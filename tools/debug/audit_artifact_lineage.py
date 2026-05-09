"""
Artifact Lineage Audit Tool

检查所有 artifacts 的血缘关系，确保它们都是基于本轮 Tushare 数据生成的。
"""

import os
import pandas as pd
import json
from datetime import datetime


def get_file_info(filepath):
    """获取文件信息"""
    if os.path.exists(filepath):
        mtime = os.path.getmtime(filepath)
        mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        size = os.path.getsize(filepath)
        
        row_count = 'N/A'
        col_count = 'N/A'
        try:
            if filepath.endswith('.parquet'):
                df = pd.read_parquet(filepath)
                row_count = len(df)
                col_count = len(df.columns)
            elif filepath.endswith('.json'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    row_count = 'object'
                    col_count = len(data.keys()) if isinstance(data, dict) else len(data)
        except Exception as e:
            row_count = f'error: {e}'
            col_count = 'error'
        
        return {
            'exists': True,
            'modified': mtime_str,
            'size_bytes': size,
            'row_count': row_count,
            'col_count': col_count,
            'mtime': mtime
        }
    return {
        'exists': False,
        'modified': 'N/A',
        'size_bytes': 0,
        'row_count': 'N/A',
        'col_count': 'N/A',
        'mtime': 0
    }


def main():
    profile = 'research_medium_trial'
    processed_dir = f'data/processed/profiles/{profile}'
    dashboard_dir = f'data/dashboard/profiles/{profile}'
    
    files_to_check = [
        ('prices.parquet', processed_dir),
        ('prices_metadata.json', processed_dir),
        ('factor_summary.parquet', dashboard_dir),
        ('factor_ic_series.parquet', dashboard_dir),
        ('factor_correlation.parquet', dashboard_dir),
        ('formula_factors.parquet', dashboard_dir),
        ('formula_factor_metadata.json', dashboard_dir),
        ('neural_factors.parquet', dashboard_dir),
        ('neural_factor_summary.parquet', dashboard_dir),
        ('neural_factor_metadata.json', dashboard_dir),
        ('oos_feature_comparison.parquet', dashboard_dir),
        ('oos_rankic_series.parquet', dashboard_dir),
        ('oos_backtest_summary.parquet', dashboard_dir),
        ('oos_equity_curves.parquet', dashboard_dir),
        ('trading_constraint_summary.parquet', dashboard_dir),
        ('tradable_mask.parquet', dashboard_dir),
        ('constrained_backtest_summary.json', dashboard_dir),
        ('constrained_equity_curve.parquet', dashboard_dir),
        ('constrained_drawdown_curve.parquet', dashboard_dir),
        ('stage_status.json', dashboard_dir),
        ('profile_manifest.json', dashboard_dir),
    ]
    
    results = {}
    prices_mtime = 0
    prices_exists = False
    
    for filename, dirpath in files_to_check:
        filepath = os.path.join(dirpath, filename)
        info = get_file_info(filepath)
        results[filename] = info
        if filename == 'prices.parquet' and info['exists']:
            prices_mtime = info['mtime']
            prices_exists = True
    
    report_lines = [
        '# Artifact Lineage Audit Report',
        '',
        f'## Profile: {profile}',
        f'## Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        '',
        '## 1. Artifact Summary',
        '',
        '| Artifact | Exists | Modified Time | Size (bytes) | Rows | Columns | Status |',
        '|----------|--------|---------------|--------------|------|---------|--------|',
    ]
    
    stale_files = []
    fresh_files = []
    missing_files = []
    
    for filename, info in results.items():
        if not info['exists']:
            status = 'MISSING'
            missing_files.append(filename)
        elif prices_exists and info['mtime'] < prices_mtime:
            status = 'STALE'
            stale_files.append(filename)
        else:
            status = 'FRESH'
            fresh_files.append(filename)
        
        report_lines.append(f'| {filename} | {info["exists"]} | {info["modified"]} | {info["size_bytes"]} | {info["row_count"]} | {info["col_count"]} | {status} |')
    
    report_lines.extend([
        '',
        '## 2. Status Summary',
        '',
        f'- **FRESH (基于本轮 prices)** : {len(fresh_files)}',
        f'- **STALE (早于本轮 prices)** : {len(stale_files)}',
        f'- **MISSING** : {len(missing_files)}',
        '',
    ])
    
    if stale_files:
        report_lines.extend([
            '## 3. STALE Artifacts (必须重跑)',
            '',
        ])
        for f in stale_files:
            report_lines.append(f'- {f}')
    
    if missing_files:
        report_lines.extend([
            '## 4. MISSING Artifacts',
            '',
        ])
        for f in missing_files:
            report_lines.append(f'- {f}')
    
    report_lines.extend([
        '',
        '## 5. Global File Check',
        '',
    ])
    
    global_dashboard = 'data/dashboard'
    has_global_issues = False
    for filename in ['factor_summary.parquet', 'formula_factors.parquet', 'neural_factors.parquet']:
        global_path = os.path.join(global_dashboard, filename)
        exists = os.path.exists(global_path)
        report_lines.append(f'- Global {filename}: {"EXISTS" if exists else "NOT FOUND"}')
        if exists:
            report_lines.append(f'  - WARNING: 全局文件存在，需确认是否被误读')
            has_global_issues = True
    
    report_lines.extend([
        '',
        '## 6. Conclusion',
        '',
    ])
    
    if stale_files or missing_files:
        report_lines.append(f'### Status: FAIL')
        report_lines.append(f'- 存在 {len(stale_files)} 个 STALE artifacts，需要重跑')
        report_lines.append(f'- 存在 {len(missing_files)} 个 MISSING artifacts')
        if has_global_issues:
            report_lines.append(f'- 存在全局文件，需确认是否被误读')
    else:
        report_lines.append(f'### Status: PASS')
        report_lines.append(f'- 所有 artifacts 都是 FRESH')
    
    os.makedirs('reports', exist_ok=True)
    with open('reports/artifact_lineage_audit_report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print('报告已生成: reports/artifact_lineage_audit_report.md')
    print(f'FRESH: {len(fresh_files)}, STALE: {len(stale_files)}, MISSING: {len(missing_files)}')
    
    return len(stale_files) + len(missing_files)


if __name__ == '__main__':
    exit(main())
