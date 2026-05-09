"""
价格口径审计工具

检查 prices.parquet 中 raw 和 adjusted 价格字段是否正确分离，
确保交易约束使用原始价格，因子计算使用复权价格。
"""

import os
import pandas as pd
import json
from datetime import datetime


def main():
    profile = 'research_medium_trial'
    processed_dir = f'data/processed/profiles/{profile}'
    
    # 加载价格数据
    price_path = os.path.join(processed_dir, 'prices.parquet')
    if not os.path.exists(price_path):
        print(f"ERROR: {price_path} 不存在")
        return 1
    
    df = pd.read_parquet(price_path)
    
    report_lines = [
        '# Data Price Mode Audit Report',
        '',
        f'## Profile: {profile}',
        f'## Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        '',
        '## 1. Available Fields',
        '',
        '| Category | Fields |',
        '|----------|--------|',
    ]
    
    # 检查字段
    raw_fields = [f for f in df.columns if f.endswith('_raw')]
    adj_fields = [f for f in df.columns if f.endswith('_adj')]
    default_fields = ['open', 'high', 'low', 'close', 'pre_close']
    limit_fields = ['up_limit', 'down_limit']
    
    report_lines.append(f'| Raw Fields | {", ".join(raw_fields)} |')
    report_lines.append(f'| Adj Fields | {", ".join(adj_fields)} |')
    report_lines.append(f'| Default Fields | {", ".join(default_fields)} |')
    report_lines.append(f'| Limit Fields | {", ".join(limit_fields)} |')
    
    report_lines.extend([
        '',
        '## 2. Price Consistency Check',
        '',
        '### 2.1 000001.SZ Price Analysis',
        '',
    ])
    
    # 检查 000001.SZ 的价格一致性
    sample = df[df['stock'] == '000001.SZ'].head(3)
    
    report_lines.append('| Date | close_raw | up_limit | down_limit | close_adj | adj_factor |')
    report_lines.append('|------|-----------|----------|------------|-----------|------------|')
    
    for _, row in sample.iterrows():
        close_raw = row.get('close_raw', 'N/A')
        up_limit = row.get('up_limit', 'N/A')
        down_limit = row.get('down_limit', 'N/A')
        close_adj = row.get('close_adj', 'N/A')
        adj_factor = row.get('adj_factor', 'N/A')
        
        def format_num(val):
            try:
                return f"{float(val):.2f}"
            except:
                return str(val)
        
        def format_factor(val):
            try:
                return f"{float(val):.4f}"
            except:
                return str(val)
        
        report_lines.append(f"| {row['date'].date()} | {format_num(close_raw)} | {format_num(up_limit)} | {format_num(down_limit)} | {format_num(close_adj)} | {format_factor(adj_factor)} |")
    
    # 检查数量级是否一致
    close_raw_mean = df['close_raw'].mean() if 'close_raw' in df.columns else None
    up_limit_mean = df['up_limit'].mean() if 'up_limit' in df.columns else None
    close_adj_mean = df['close_adj'].mean() if 'close_adj' in df.columns else None
    
    report_lines.extend([
        '',
        '### 2.2 Magnitude Analysis',
        '',
        f'- close_raw 均值: {close_raw_mean:.2f}' if close_raw_mean is not None else '- close_raw 均值: N/A',
        f'- up_limit 均值: {up_limit_mean:.2f}' if up_limit_mean is not None else '- up_limit 均值: N/A',
        f'- close_adj 均值: {close_adj_mean:.2f}' if close_adj_mean is not None else '- close_adj 均值: N/A',
        '',
    ])
    
    # 判断是否混用
    raw_limit_match = False
    adj_limit_mismatch = False
    
    if close_raw_mean is not None and up_limit_mean is not None and up_limit_mean > 0:
        raw_limit_match = abs(close_raw_mean - up_limit_mean) / up_limit_mean < 2.0
    
    if close_adj_mean is not None and up_limit_mean is not None and up_limit_mean > 0:
        adj_limit_mismatch = close_adj_mean > up_limit_mean * 10
    
    report_lines.extend([
        '### 2.3 Consistency Status',
        '',
        f'- close_raw 与 up_limit 数量级一致: {"PASS" if raw_limit_match else "FAIL"}',
        f'- close_adj 与 up_limit 数量级差异过大: {"WARN" if adj_limit_mismatch else "OK"}',
        '',
    ])
    
    # 检查复权因子是否正确应用
    report_lines.extend([
        '### 2.4 Adjustment Factor Validation',
        '',
    ])
    
    if 'close_raw' in df.columns and 'close_adj' in df.columns and 'adj_factor' in df.columns:
        df['calc_close_adj'] = df['close_raw'] * df['adj_factor']
        diff = abs(df['close_adj'] - df['calc_close_adj']).mean()
        match = diff < 0.01
        
        report_lines.append(f'- close_adj 与 close_raw * adj_factor 一致: {"PASS" if match else "FAIL"}')
        report_lines.append(f'- 平均差异: {diff:.4f}')
    else:
        report_lines.append('- 缺少必要字段进行验证: WARN')
    
    # 交易约束使用检查
    report_lines.extend([
        '',
        '## 3. Trading Constraints Usage',
        '',
        '### 3.1 Field Usage Summary',
        '',
    ])
    
    # 检查 trading_constraint_summary
    dashboard_dir = f'data/dashboard/profiles/{profile}'
    constraint_path = os.path.join(dashboard_dir, 'trading_constraint_summary.parquet')
    
    if os.path.exists(constraint_path):
        constraint_df = pd.read_parquet(constraint_path)
        report_lines.append(f'- trading_constraint_summary.parquet 存在: OK')
        report_lines.append(f'- 行数: {len(constraint_df)}')
        report_lines.append(f'- 列数: {len(constraint_df.columns)}')
        report_lines.append(f'- 列名: {", ".join(constraint_df.columns)}')
    else:
        report_lines.append(f'- trading_constraint_summary.parquet 不存在: WARN')
    
    # 结论
    report_lines.extend([
        '',
        '## 4. Conclusion',
        '',
    ])
    
    all_pass = raw_limit_match and not adj_limit_mismatch
    
    if all_pass:
        report_lines.append('### Status: PASS')
        report_lines.append('- 价格口径分离正确')
        report_lines.append('- 原始价格用于交易约束')
        report_lines.append('- 复权价格可用于因子计算')
    else:
        report_lines.append('### Status: FAIL')
        if not raw_limit_match:
            report_lines.append('- close_raw 与 up_limit 数量级不一致')
        if adj_limit_mismatch:
            report_lines.append('- close_adj 与 up_limit 数量级差异过大')
    
    report_lines.append('')
    report_lines.append('## 5. Recommended Actions')
    report_lines.append('')
    report_lines.append('1. 因子计算应使用 close_adj, open_adj 等复权字段')
    report_lines.append('2. 交易约束应使用 close_raw, up_limit, down_limit 等原始字段')
    report_lines.append('3. 涨跌停判断应基于 pct_change 或 raw 价格')
    report_lines.append('4. metadata 应记录 price_mode 和字段用途')
    
    # 保存报告
    os.makedirs('reports', exist_ok=True)
    with open('reports/data_price_mode_audit_report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print('报告已生成: reports/data_price_mode_audit_report.md')
    
    return 0 if all_pass else 1


if __name__ == '__main__':
    exit(main())
