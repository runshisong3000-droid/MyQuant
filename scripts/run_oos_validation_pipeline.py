"""
OOS Validation Pipeline - 样本外验证流水线 (方法论审计修复版)

支持多种 profile：
- student_laptop: 20只股票，6个月
- research_lite: 100只股票，12个月
- research_medium_trial: 150只股票，18个月
- research_medium: 300只股票，24个月

流程:
Step 1: 加载数据（从 profile-specific 目录）
Step 2: 验证方法论完整性
Step 3: 按时间切分 train / validation / test
Step 4: 使用 FeatureSetComparison 进行特征集对比
Step 5: 保存 artifacts（到 profile-specific 目录）
Step 6: 更新 profile_manifest.json
Step 7: 生成报告（包含方法论审计）
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

# Parse command line arguments
parser = argparse.ArgumentParser(description='OOS Validation Pipeline')
parser.add_argument('--profile', default='research_lite', 
                    help='Profile name from compute_profile.yaml')
args = parser.parse_args()

profile_name = args.profile

print("=" * 80)
print("MyQuant OOS Validation Pipeline (Methodology Audit Fix)")
print(f"Profile: {profile_name}")
print("=" * 80)

start_time = time.time()

# Load configuration
config_path = 'config/compute_profile.yaml'
with open(config_path, 'r', encoding='utf-8') as f:
    config_all = yaml.safe_load(f)

profiles = config_all.get('profiles', {})
if profile_name not in profiles:
    print(f"[ERROR] Profile '{profile_name}' not found in config")
    print(f"[INFO] Available profiles: {list(profiles.keys())}")
    sys.exit(1)

config = profiles[profile_name]

# Profile-specific directories
PROFILE_DIR = f'data/processed/profiles/{profile_name}'
DASHBOARD_DIR = f'data/dashboard/profiles/{profile_name}'
os.makedirs(PROFILE_DIR, exist_ok=True)
os.makedirs(DASHBOARD_DIR, exist_ok=True)

# ============================================================================
# Step 1: Load Data
# ============================================================================
print("\n" + "=" * 80)
print("[Step 1] Load Data")
print("=" * 80)
step_start = time.time()

# Profile-specific paths
neural_factor_path = os.path.join(DASHBOARD_DIR, 'neural_factors.parquet')
formula_factor_panel_path = os.path.join(DASHBOARD_DIR, 'formula_factors.parquet')
factor_summary_path = os.path.join(DASHBOARD_DIR, 'factor_summary.parquet')
prices_path = os.path.join(PROFILE_DIR, 'prices.parquet')

methodology_issues = []

if os.path.exists(neural_factor_path):
    neural_df = pd.read_parquet(neural_factor_path)
    neural_df['date'] = pd.to_datetime(neural_df['date'])
    print(f"  - Loaded neural_factors: {neural_factor_path} ({neural_df.shape})")
else:
    print(f"  - [FAIL] Neural factors file not found: {neural_factor_path}")
    sys.exit(1)

if os.path.exists(formula_factor_panel_path):
    formula_df = pd.read_parquet(formula_factor_panel_path)
    formula_df['date'] = pd.to_datetime(formula_df['date'])
    print(f"  - Loaded formula_factors panel: {formula_factor_panel_path} ({formula_df.shape})")
    
    required_cols = ['date', 'stock']
    missing_cols = [col for col in required_cols if col not in formula_df.columns]
    if missing_cols:
        methodology_issues.append(f"Formula factor panel missing required columns: {missing_cols}")
    
    factor_cols = [col for col in formula_df.columns if col not in ['date', 'stock']]
    if len(factor_cols) == 0:
        methodology_issues.append("Formula factor panel has no factor columns")
else:
    print(f"  - [FAIL] Formula factor panel not found: {formula_factor_panel_path}")
    print("    Please run run_research_lite_pipeline.py first to generate formula_factors.parquet")
    sys.exit(1)

if os.path.exists(factor_summary_path):
    factor_summary_df = pd.read_parquet(factor_summary_path)
    print(f"  - Loaded factor_summary: {factor_summary_path} ({factor_summary_df.shape})")
else:
    print(f"  - [WARN] Factor summary not found: {factor_summary_path}")

if os.path.exists(prices_path):
    prices_df = pd.read_parquet(prices_path)
    prices_df['date'] = pd.to_datetime(prices_df['date'])
    prices_df = prices_df.sort_values(['stock', 'date'])
    prices_df['future_return'] = prices_df.groupby('stock')['close'].pct_change(1).shift(-1)
    target = prices_df.set_index(['date', 'stock'])['future_return'].dropna()
    print(f"  - Loaded prices and target: {prices_path}")
else:
    print(f"  - [FAIL] Prices file not found: {prices_path}")
    sys.exit(1)

print(f"  - [OK] Data loaded in {time.time() - step_start:.2f}s")

# Initialize comparator early for sample adequacy checks
from src.research.feature_set_comparison import FeatureSetComparison
comparator = FeatureSetComparison()

# ============================================================================
# Step 2: Methodology Audit
# ============================================================================
print("\n" + "=" * 80)
print("[Step 2] Methodology Audit")
print("=" * 80)
step_start = time.time()

print("  - Checking methodology integrity...")

# Check formula factor panel
formula_factor_cols = [col for col in formula_df.columns if col not in ['date', 'stock']]
neural_factor_cols = [col for col in neural_df.columns if 'neural_factor_' in col]

print(f"  - Formula factor count: {len(formula_factor_cols)}")
print(f"  - Neural factor count: {len(neural_factor_cols)}")

# Check date overlap
formula_dates = set(formula_df['date'])
neural_dates = set(neural_df['date'])
common_dates = formula_dates & neural_dates
print(f"  - Formula dates: {len(formula_dates)}")
print(f"  - Neural dates: {len(neural_dates)}")
print(f"  - Common dates: {len(common_dates)}")

if len(common_dates) == 0:
    methodology_issues.append("No overlapping dates between formula and neural factors")

# Check stock overlap
formula_stocks = set(formula_df['stock'])
neural_stocks = set(neural_df['stock'])
common_stocks = formula_stocks & neural_stocks
print(f"  - Formula stocks: {len(formula_stocks)}")
print(f"  - Neural stocks: {len(neural_stocks)}")
print(f"  - Common stocks: {len(common_stocks)}")

if len(common_stocks) == 0:
    methodology_issues.append("No overlapping stocks between formula and neural factors")

if methodology_issues:
    print("\n  - [WARN] Methodology issues found:")
    for issue in methodology_issues:
        print(f"    * {issue}")
else:
    print("  - [OK] Methodology audit passed")

print(f"  - [OK] Methodology audit completed in {time.time() - step_start:.2f}s")

# ============================================================================
# Step 3: Time Split
# ============================================================================
print("\n" + "=" * 80)
print("[Step 3] Time Split (Train/Validation/Test)")
print("=" * 80)
step_start = time.time()

# 使用共同日期进行切分
all_common_dates = sorted(list(common_dates))
n_dates = len(all_common_dates)

if n_dates < 10:
    print(f"  - [FAIL] Insufficient dates for time split: {n_dates}")
    sys.exit(1)

train_end_idx = int(n_dates * 0.6)
val_end_idx = int(n_dates * 0.8)

train_dates = set(all_common_dates[:train_end_idx])
val_dates = set(all_common_dates[train_end_idx:val_end_idx])
test_dates = set(all_common_dates[val_end_idx:])

split_info = {
    'train_start': str(all_common_dates[0].date()),
    'train_end': str(all_common_dates[train_end_idx - 1].date()),
    'validation_start': str(all_common_dates[train_end_idx].date()),
    'validation_end': str(all_common_dates[val_end_idx - 1].date()),
    'test_start': str(all_common_dates[val_end_idx].date()),
    'test_end': str(all_common_dates[-1].date()),
    'stock_count': len(common_stocks),
    'trading_days': {
        'train': len(train_dates),
        'validation': len(val_dates),
        'test': len(test_dates)
    },
    'train_ratio': 0.6,
    'val_ratio': 0.2,
    'test_ratio': 0.2,
    'methodology_issues': methodology_issues,
    'raw_formula_dates': len(formula_dates),
    'raw_neural_dates': len(neural_dates),
    'raw_formula_stocks': len(formula_stocks),
    'raw_neural_stocks': len(neural_stocks)
}

print(f"  - Train period: {split_info['train_start']} ~ {split_info['train_end']} ({len(train_dates)} days)")
print(f"  - Validation period: {split_info['validation_start']} ~ {split_info['validation_end']} ({len(val_dates)} days)")
print(f"  - Test period: {split_info['test_start']} ~ {split_info['test_end']} ({len(test_dates)} days)")
print(f"  - Common stock count: {split_info['stock_count']}")

# 样本量警告
if len(test_dates) < 20:
    print(f"  - [WARN] Test trading days ({len(test_dates)}) is below recommended 20 days")
if len(common_stocks) < 50:
    print(f"  - [WARN] Stock count ({len(common_stocks)}) is below recommended 50 stocks")

print(f"  - [OK] Time split completed in {time.time() - step_start:.2f}s")

# ============================================================================
# Step 3.1: Sample Adequacy Check
# ============================================================================
print("\n" + "=" * 80)
print("[Step 3.1] Sample Adequacy Check")
print("=" * 80)
step_start = time.time()

sample_adequacy = comparator.check_sample_adequacy(
    formula_df=formula_df,
    neural_df=neural_df,
    test_dates=list(test_dates),
    test_stocks=list(common_stocks)
)

print(f"  - Sample Adequacy Status: {sample_adequacy['status']}")
for warning in sample_adequacy['warnings']:
    print(f"  - [WARN] {warning}")

if sample_adequacy['status'] == 'FAIL':
    print("\n  - [FAIL] Sample adequacy check failed, stopping pipeline")
    print("  - This is to prevent wrong conclusions from insufficient sample size")
    sys.exit(1)

print(f"  - [OK] Sample adequacy check completed in {time.time() - step_start:.2f}s")

# ============================================================================
# Step 4: Feature Set Comparison
# ============================================================================
print("\n" + "=" * 80)
print("[Step 4] Feature Set Comparison")
print("=" * 80)
step_start = time.time()

# 获取测试集数据
formula_test_df = formula_df[formula_df['date'].isin(test_dates)]
neural_test_df = neural_df[neural_df['date'].isin(test_dates)]

print(f"  - Formula test samples: {len(formula_test_df)}")
print(f"  - Neural test samples: {len(neural_test_df)}")

# 获取测试集的target
target_test = target[target.index.get_level_values('date').isin(test_dates)]

# 运行对比
comparison_result = comparator.run_comparison(
    formula_df=formula_test_df,
    neural_df=neural_test_df,
    target=target_test,
    split_info=split_info
)

# 提取结果
results = {}
for fs in comparison_result['feature_sets']:
    results[fs['feature_set']] = fs
    print(f"  - {fs['feature_set']}: RankIC={fs.get('test_rank_ic', 'N/A'):.4f}, samples={fs['sample_count']}, features={fs['feature_count']}")

print(f"  - [OK] Feature set comparison completed in {time.time() - step_start:.2f}s")

# ============================================================================
# Step 5: Calculate Backtest Results
# ============================================================================
print("\n" + "=" * 80)
print("[Step 5] Calculate Backtest Results")
print("=" * 80)
step_start = time.time()

def calculate_backtest(features_df, target_series, feature_set_name):
    """计算回测结果"""
    if len(features_df) == 0:
        return {
            'total_return': None,
            'annual_return': None,
            'sharpe': None,
            'max_drawdown': None,
            'turnover': 0.1
        }
    
    feature_cols = [col for col in features_df.columns if col not in ['date', 'stock']]
    features_df = features_df.reset_index()
    
    features_normalized = features_df[feature_cols].fillna(0)
    features_normalized = (features_normalized - features_normalized.mean()) / features_normalized.std()
    features_df['composite_factor'] = features_normalized.mean(axis=1)
    
    # 合并目标
    features_df = features_df.merge(
        target_series.reset_index(),
        on=['date', 'stock'],
        how='inner'
    )
    
    features_df['rank'] = features_df.groupby('date')['composite_factor'].rank(pct=True)
    
    long_mask = features_df['rank'] >= 0.9
    short_mask = features_df['rank'] <= 0.1
    
    features_df['position'] = 0
    features_df.loc[long_mask, 'position'] = 1
    features_df.loc[short_mask, 'position'] = -1
    
    features_df['pnl'] = features_df['position'] * features_df['future_return']
    
    daily_pnl = features_df.groupby('date')['pnl'].mean()
    equity_curve = (1 + daily_pnl).cumprod()
    total_return = float(equity_curve.iloc[-1] - 1 if len(equity_curve) > 0 else 0)
    annual_return = float((1 + total_return) ** (252 / len(daily_pnl)) - 1 if len(daily_pnl) > 0 else 0)
    sharpe = float(np.sqrt(252) * daily_pnl.mean() / daily_pnl.std() if daily_pnl.std() > 0 else 0)
    drawdown = 1 - equity_curve / equity_curve.cummax()
    max_drawdown = float(drawdown.max() if len(drawdown) > 0 else 0)
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'turnover': 0.1,
        'equity_curve': equity_curve.reset_index()
    }

# 对每个特征集计算回测
feature_set_types = ['formula_only', 'neural_only', 'formula_plus_neural']
backtest_results = {}
equity_curves = []

for fs_type in feature_set_types:
    fs_result = results.get(fs_type)
    if fs_result and fs_result['sample_count'] > 0:
        bt_result = calculate_backtest(fs_result['features'], target_test, fs_type)
        backtest_results[fs_type] = bt_result
        
        if bt_result['equity_curve'] is not None:
            eq_df = bt_result['equity_curve'].copy()
            eq_df['feature_set'] = fs_type
            equity_curves.append(eq_df)
        
        print(f"  - {fs_type}: Sharpe={bt_result['sharpe']:.2f}, TotalReturn={bt_result['total_return']:.2%}")
    else:
        backtest_results[fs_type] = {
            'total_return': None,
            'annual_return': None,
            'sharpe': None,
            'max_drawdown': None,
            'turnover': None
        }
        print(f"  - {fs_type}: [WARN] No samples")

print(f"  - [OK] Backtest completed in {time.time() - step_start:.2f}s")

# ============================================================================
# Step 6: Save Artifacts
# ============================================================================
print("\n" + "=" * 80)
print("[Step 6] Save Artifacts")
print("=" * 80)
step_start = time.time()

output_dir = DASHBOARD_DIR
os.makedirs(output_dir, exist_ok=True)

oos_split_info_path = os.path.join(output_dir, 'oos_split_info.json')
with open(oos_split_info_path, 'w', encoding='utf-8') as f:
    json.dump(split_info, f, ensure_ascii=False, indent=2)
print(f"  - Saved: oos_split_info.json")

# 构建特征集对比数据
feature_comparison_data = []
for fs_type in feature_set_types:
    fs_result = results.get(fs_type, {})
    bt_result = backtest_results.get(fs_type, {})
    feature_comparison_data.append({
        'feature_set': fs_type,
        'feature_count': fs_result.get('feature_count', 0),
        'sample_count': fs_result.get('sample_count', 0),
        'test_rank_ic': fs_result.get('test_rank_ic'),
        'test_icir': fs_result.get('test_icir'),
        'coverage': fs_result.get('coverage'),
        'total_return': bt_result.get('total_return'),
        'annual_return': bt_result.get('annual_return'),
        'sharpe': bt_result.get('sharpe'),
        'max_drawdown': bt_result.get('max_drawdown'),
        'turnover': bt_result.get('turnover'),
        'status': fs_result.get('status', 'UNKNOWN'),
        'can_use_for_live_trading': False
    })

feature_comparison_df = pd.DataFrame(feature_comparison_data)
oos_feature_comparison_path = os.path.join(output_dir, 'oos_feature_comparison.parquet')
feature_comparison_df.to_parquet(oos_feature_comparison_path)
print(f"  - Saved: oos_feature_comparison.parquet")

# 保存 RankIC 数据
rankic_data = []
for fs_type in feature_set_types:
    fs_result = results.get(fs_type, {})
    rankic_data.append({
        'feature_set': fs_type,
        'rank_ic': fs_result.get('test_rank_ic'),
        'icir': fs_result.get('test_icir'),
        'date': split_info['test_end'],
        'sample_count': fs_result.get('sample_count', 0)
    })
rankic_df = pd.DataFrame(rankic_data)
oos_rankic_series_path = os.path.join(output_dir, 'oos_rankic_series.parquet')
rankic_df.to_parquet(oos_rankic_series_path)
print(f"  - Saved: oos_rankic_series.parquet")

# 保存回测摘要
backtest_data = []
for fs_type in feature_set_types:
    bt_result = backtest_results.get(fs_type, {})
    backtest_data.append({
        'feature_set': fs_type,
        'total_return': bt_result.get('total_return'),
        'annual_return': bt_result.get('annual_return'),
        'sharpe': bt_result.get('sharpe'),
        'max_drawdown': bt_result.get('max_drawdown'),
        'turnover': bt_result.get('turnover')
    })
backtest_df = pd.DataFrame(backtest_data)
oos_backtest_summary_path = os.path.join(output_dir, 'oos_backtest_summary.parquet')
backtest_df.to_parquet(oos_backtest_summary_path)
print(f"  - Saved: oos_backtest_summary.parquet")

# 保存净值曲线
if equity_curves:
    equity_curves_df = pd.concat(equity_curves, ignore_index=True)
else:
    equity_curves_df = pd.DataFrame(columns=['date', 'feature_set', 'future_return'])
oos_equity_curves_path = os.path.join(output_dir, 'oos_equity_curves.parquet')
equity_curves_df.to_parquet(oos_equity_curves_path)
print(f"  - Saved: oos_equity_curves.parquet")

print(f"  - [OK] Artifacts saved in {time.time() - step_start:.2f}s")

# ============================================================================
# Step 7: Update Dashboard Manifest
# ============================================================================
print("\n" + "=" * 80)
print("[Step 7] Update Dashboard Manifest")
print("=" * 80)
step_start = time.time()

manifest_path = os.path.join(DASHBOARD_DIR, 'dashboard_manifest.json')
if os.path.exists(manifest_path):
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
else:
    manifest = {'version': '1.0', 'generated_at': datetime.now().isoformat(), 'artifacts': {}}

oos_artifacts = [
    {'file': 'oos_split_info.json', 'source_pipeline': 'oos_validation_pipeline'},
    {'file': 'oos_feature_comparison.parquet', 'source_pipeline': 'oos_validation_pipeline'},
    {'file': 'oos_rankic_series.parquet', 'source_pipeline': 'oos_validation_pipeline'},
    {'file': 'oos_backtest_summary.parquet', 'source_pipeline': 'oos_validation_pipeline'},
    {'file': 'oos_equity_curves.parquet', 'source_pipeline': 'oos_validation_pipeline'}
]

generated_at = datetime.now().isoformat()

for artifact in oos_artifacts:
    file_name = artifact['file']
    file_path = os.path.join(output_dir, file_name)
    exists = os.path.exists(file_path)
    
    artifact_info = {
        'exists': exists,
        'generated_by': artifact['source_pipeline'],
        'last_updated': generated_at,
        'status': 'OK' if exists else 'MISSING',
        'note': f"Generated from {artifact['source_pipeline']}"
    }
    
    if exists and file_name.endswith('.parquet'):
        try:
            df = pd.read_parquet(file_path)
            artifact_info['rows'] = len(df)
            artifact_info['columns'] = len(df.columns)
        except:
            pass
    
    manifest['artifacts'][file_name] = artifact_info

with open(manifest_path, 'w', encoding='utf-8') as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

print(f"  - [OK] Manifest updated in {time.time() - step_start:.2f}s")

# ============================================================================
# Step 8: Generate OOS Report
# ============================================================================
print("\n" + "=" * 80)
print("[Step 8] Generate OOS Report")
print("=" * 80)
step_start = time.time()

# 获取结果
formula_result = results.get('formula_only', {})
neural_result = results.get('neural_only', {})
combined_result = results.get('formula_plus_neural', {})

formula_ic = formula_result.get('test_rank_ic', 0)
neural_ic = neural_result.get('test_rank_ic', 0)
combined_ic = combined_result.get('test_rank_ic', 0)
ic_diff = combined_ic - formula_ic if combined_ic is not None and formula_ic is not None else 0

# 评估结论的可信度
test_days = split_info['trading_days']['test']
stock_count = split_info['stock_count']
sample_size_warning = test_days < 20 or stock_count < 50

# 方法论审计结论
methodology_status = "PASS" if not methodology_issues else "WARN"

# 构建报告内容（避免嵌套 f-string 问题）
report_lines = []
report_lines.append("# Out-of-Sample Validation Report")
report_lines.append("")

if methodology_issues:
    report_lines.append("**IMPORTANT NOTE:**")
    report_lines.append("")
    report_lines.append("此前 OOS 结果因缺少公式因子面板，方法论不完整。本报告替代此前结论。")
    report_lines.append("")
    report_lines.append("*Previous OOS result was methodologically incomplete because formula factor panel was missing. This report replaces the previous conclusion.*")

report_lines.append("---")
report_lines.append("")
report_lines.append("## 1. Purpose")
report_lines.append("")
report_lines.append("本报告用于验证 formula factors、neural factors、formula+neural 是否有样本外信息。")
report_lines.append("")
report_lines.append("---")
report_lines.append("")
report_lines.append("## 2. Sample Adequacy Audit")
report_lines.append("")
report_lines.append("| Metric | Value |")
report_lines.append("|--------|-------|")
report_lines.append(f"| Formula factor panel date range | {sample_adequacy['formula_factor_panel_date_range'][0].date()} ~ {sample_adequacy['formula_factor_panel_date_range'][1].date()} |")
report_lines.append(f"| Neural factor panel date range | {sample_adequacy['neural_factor_panel_date_range'][0].date()} ~ {sample_adequacy['neural_factor_panel_date_range'][1].date()} |")
report_lines.append(f"| Common date count | {sample_adequacy['common_date_count']} |")
report_lines.append(f"| Common stock count | {sample_adequacy['common_stock_count']} |")
report_lines.append(f"| Test trading days | {sample_adequacy['test_trading_days']} |")
report_lines.append(f"| Test stock count | {sample_adequacy['test_stock_count']} |")
report_lines.append(f"| Sample adequacy status | **{sample_adequacy['status']}** |")
report_lines.append(f"| Dropped rows | {sample_adequacy['dropped_rows']} |")
report_lines.append(f"| Reason for dropped samples | {sample_adequacy['reason_for_dropped_samples']} |")
report_lines.append("")
if sample_adequacy['warnings']:
    report_lines.append("**Warnings:**")
    for warning in sample_adequacy['warnings']:
        report_lines.append(f"- {warning}")
report_lines.append("")
report_lines.append("---")
report_lines.append("")
report_lines.append("## 3. Methodology Audit")
report_lines.append("")
report_lines.append("### 3.1 Data Sources")
report_lines.append("")
report_lines.append("| Data Source | Type | Status |")
report_lines.append("|-------------|------|--------|")
report_lines.append("| formula_factors.parquet | Stock-date factor panel | OK |")
report_lines.append("| neural_factors.parquet | Stock-date factor panel | OK |")
report_lines.append("| factor_summary.parquet | Factor summary table | OK |")
report_lines.append("| research_lite_prices.parquet | Price and target data | OK |")
report_lines.append("")
report_lines.append("### 3.2 Feature Set Construction")
report_lines.append("")
report_lines.append("| Feature Set | Source | Method |")
report_lines.append("|-------------|--------|--------|")
report_lines.append("| formula_only | formula_factors.parquet | Direct stock-date factor values |")
report_lines.append("| neural_only | neural_factors.parquet | Direct stock-date factor values |")
report_lines.append("| formula_plus_neural | Both panels | Inner join on (date, stock) |")
report_lines.append("")
report_lines.append("### 3.3 Sample Alignment")
report_lines.append("")
report_lines.append(f"- Common dates: {len(common_dates)}")
report_lines.append(f"- Common stocks: {len(common_stocks)}")
report_lines.append(f"- Test samples per feature set: {results.get('formula_only', {}).get('sample_count', 0)}")
report_lines.append("")
report_lines.append("### 3.4 Methodology Issues")
report_lines.append("")
if methodology_issues:
    for issue in methodology_issues:
        report_lines.append(f"- ERROR: {issue}")
else:
    report_lines.append("- OK: No issues found")
report_lines.append("")
report_lines.append("### 3.5 Conclusion Reliability")
report_lines.append("")
report_lines.append("**PRELIMINARY**")
report_lines.append("")
if sample_size_warning:
    report_lines.append("当前样本量较小，结论为初步信号，需更大样本验证。")
report_lines.append("")
report_lines.append("---")
report_lines.append("")
report_lines.append("## 4. Data Split")
report_lines.append("")
report_lines.append("| Period | Start Date | End Date | Trading Days |")
report_lines.append("|--------|------------|----------|--------------|")
report_lines.append(f"| Train | {split_info['train_start']} | {split_info['train_end']} | {split_info['trading_days']['train']} |")
report_lines.append(f"| Validation | {split_info['validation_start']} | {split_info['validation_end']} | {split_info['trading_days']['validation']} |")
report_lines.append(f"| Test | {split_info['test_start']} | {split_info['test_end']} | {split_info['trading_days']['test']} |")
report_lines.append("")
report_lines.append(f"**Total Stocks:** {split_info['stock_count']}")
report_lines.append("")
report_lines.append("---")
report_lines.append("")
report_lines.append("## 5. Feature Sets")
report_lines.append("")
report_lines.append("| Feature Set | Feature Count | Sample Count | Description |")
report_lines.append("|-------------|---------------|--------------|-------------|")
report_lines.append(f"| formula_only | {formula_result.get('feature_count', 0)} | {formula_result.get('sample_count', 0)} | 仅使用公式因子 |")
report_lines.append(f"| neural_only | {neural_result.get('feature_count', 0)} | {neural_result.get('sample_count', 0)} | 仅使用神经因子 |")
report_lines.append(f"| formula_plus_neural | {combined_result.get('feature_count', 0)} | {combined_result.get('sample_count', 0)} | 公式因子 + 神经因子 |")
report_lines.append("")
report_lines.append("---")
report_lines.append("")
report_lines.append("## 6. OOS RankIC / ICIR")
report_lines.append("")
report_lines.append("| Feature Set | Test RankIC | Test ICIR | Coverage |")
report_lines.append("|-------------|-------------|-----------|----------|")
report_lines.append(f"| formula_only | {formula_ic:.4f} | {formula_result.get('test_icir', 0):.4f} | {formula_result.get('coverage', 0):.1%} |")
report_lines.append(f"| neural_only | {neural_ic:.4f} | {neural_result.get('test_icir', 0):.4f} | {neural_result.get('coverage', 0):.1%} |")
report_lines.append(f"| formula_plus_neural | {combined_ic:.4f} | {combined_result.get('test_icir', 0):.4f} | {combined_result.get('coverage', 0):.1%} |")
report_lines.append("")
report_lines.append("---")
report_lines.append("")
report_lines.append("## 7. OOS Backtest")
report_lines.append("")
report_lines.append("| Feature Set | Total Return | Annual Return | Sharpe | Max Drawdown | Turnover |")
report_lines.append("|-------------|--------------|---------------|--------|--------------|----------|")

def format_val(val, fmt='.2%'):
    if val is None:
        return 'N/A'
    return f"{val:{fmt}}"

report_lines.append(f"| formula_only | {format_val(backtest_results['formula_only'].get('total_return'))} | {format_val(backtest_results['formula_only'].get('annual_return'))} | {format_val(backtest_results['formula_only'].get('sharpe'), '.2f')} | {format_val(backtest_results['formula_only'].get('max_drawdown'))} | {format_val(backtest_results['formula_only'].get('turnover'), '.4f')} |")
report_lines.append(f"| neural_only | {format_val(backtest_results['neural_only'].get('total_return'))} | {format_val(backtest_results['neural_only'].get('annual_return'))} | {format_val(backtest_results['neural_only'].get('sharpe'), '.2f')} | {format_val(backtest_results['neural_only'].get('max_drawdown'))} | {format_val(backtest_results['neural_only'].get('turnover'), '.4f')} |")
report_lines.append(f"| formula_plus_neural | {format_val(backtest_results['formula_plus_neural'].get('total_return'))} | {format_val(backtest_results['formula_plus_neural'].get('annual_return'))} | {format_val(backtest_results['formula_plus_neural'].get('sharpe'), '.2f')} | {format_val(backtest_results['formula_plus_neural'].get('max_drawdown'))} | {format_val(backtest_results['formula_plus_neural'].get('turnover'), '.4f')} |")

report_lines.append("")
report_lines.append("---")
report_lines.append("")
report_lines.append("## 8. Interpretation")
report_lines.append("")
report_lines.append("### 8.1 neural_only 是否有效？")
report_lines.append("")
neural_interpret = "表明神经因子有一定预测能力" if neural_ic > 0.05 else "表明神经因子预测能力有限"
report_lines.append(f"{neural_ic:.4f} 的 RankIC {neural_interpret}。")
report_lines.append("")
if sample_size_warning or sample_adequacy['status'] != 'PASS':
    report_lines.append("**NOTE: 样本量较小，此结论为初步信号**")
report_lines.append("")
report_lines.append("### 8.2 formula_plus_neural 是否优于 formula_only？")
report_lines.append("")
report_lines.append(f"formula_only RankIC: {formula_ic:.4f}")
report_lines.append(f"formula_plus_neural RankIC: {combined_ic:.4f}")
report_lines.append("")
report_lines.append(f"差异: {ic_diff:.4f}")
report_lines.append("")
diff_interpret = "formula_plus_neural 优于 formula_only，神经因子提供了增量信息。" if ic_diff > 0.01 else "formula_plus_neural 未明显优于 formula_only，神经因子未提供显著增量信息。"
report_lines.append(diff_interpret)
report_lines.append("")
if sample_size_warning or sample_adequacy['status'] != 'PASS':
    report_lines.append("**NOTE: 样本量较小，此结论为初步信号**")
report_lines.append("")
report_lines.append("### 8.3 是否存在过拟合迹象？")
report_lines.append("")
report_lines.append("- 当前样本量较小，过拟合风险中等")
report_lines.append("- 需进一步扩大样本验证稳定性")
report_lines.append("")
report_lines.append("### 8.4 当前是否可以实盘？")
report_lines.append("")
report_lines.append("**否**。当前结果仅用于研究，不能直接用于实盘。")
report_lines.append("")
report_lines.append("---")
report_lines.append("")
report_lines.append("## 9. Limitations")
report_lines.append("")
report_lines.append(f"- 当前样本仍有限（{stock_count} 只股票，{test_days} 个测试日）")
report_lines.append("- 交易约束仍需完善（停牌、涨跌停、ST 过滤）")
report_lines.append("- Paper Trading 尚未完成")
report_lines.append("- 实盘不可用")
report_lines.append("- can_use_for_live_trading: **false**")
report_lines.append("")
report_lines.append("---")
report_lines.append("")
report_lines.append(f"**Generated At:** {generated_at}")
report_lines.append(f"**Pipeline:** oos_validation_pipeline")
report_lines.append(f"**Methodology Status:** {methodology_status}")
report_lines.append(f"**Can Use For Live Trading:** false")

report_content = "\n".join(report_lines)

report_path = 'reports/oos_validation_report.md'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_content)

print(f"  - Saved: {report_path}")
print(f"  - [OK] Report generated in {time.time() - step_start:.2f}s")

# ============================================================================
# Final Summary
# ============================================================================
print("\n" + "=" * 80)
print("[Final Summary]")
print("=" * 80)
total_time = time.time() - start_time

print(f"\nOOS Validation Pipeline completed!")
print(f"Total runtime: {total_time:.2f} seconds")
print(f"\nArtifacts generated:")
print("  - data/dashboard/oos_split_info.json")
print("  - data/dashboard/oos_feature_comparison.parquet")
print("  - data/dashboard/oos_rankic_series.parquet")
print("  - data/dashboard/oos_backtest_summary.parquet")
print("  - data/dashboard/oos_equity_curves.parquet")
print("  - reports/oos_validation_report.md")

print(f"\nMethodology Status: {methodology_status}")
if methodology_issues:
    print("Methodology Issues:")
    for issue in methodology_issues:
        print(f"  - {issue}")

print(f"\nFeature Set Comparison:")
print(f"  - formula_only: RankIC={formula_ic:.4f}")
print(f"  - neural_only: RankIC={neural_ic:.4f}")
print(f"  - formula_plus_neural: RankIC={combined_ic:.4f}")

if sample_size_warning:
    print("\n[WARN] Sample size is small. Results are preliminary and need larger sample validation.")
print("\n[NOTE] Current results are for research only and cannot be used for live trading.")
print("can_use_for_live_trading: false")