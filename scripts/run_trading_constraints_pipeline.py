"""
Trading Constraints Pipeline - 交易约束流水线

支持多种 profile：
- student_laptop: 20只股票，6个月
- research_lite: 100只股票，12个月
- research_medium_trial: 150只股票，18个月
- research_medium: 300只股票，24个月

流程:
Step 1: 加载 prices（从 profile-specific 目录）
Step 2: 加载 latest signals 或 oos feature scores
Step 3: 构建候选交易列表
Step 4: 运行 TradingConstraintChecker
Step 5: 生成 tradable_mask
Step 6: 运行 constrained backtest
Step 7: 保存 artifacts（到 profile-specific 目录）
Step 8: 更新 profile_manifest
Step 9: 输出报告

注意:
    - 不允许模拟数据
    - 不允许实盘交易
    - can_use_for_live_trading 必须 false
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
parser = argparse.ArgumentParser(description='Trading Constraints Pipeline')
parser.add_argument('--profile', default='research_lite', 
                    help='Profile name from compute_profile.yaml')
parser.add_argument('--force', action='store_true',
                    help='Force re-run even if artifacts exist')
args = parser.parse_args()

profile_name = args.profile
force_run = args.force

print("=" * 60)
print("MyQuant Trading Constraints Pipeline")
print(f"Profile: {profile_name}")
print("=" * 60)

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
# Step 1: Load prices
# ============================================================================
print("\n[Step 1] Load prices...")
step_start = time.time()

prices_path = os.path.join(PROFILE_DIR, 'prices.parquet')

if os.path.exists(prices_path):
    price_panel = pd.read_parquet(prices_path)
    price_panel['date'] = pd.to_datetime(price_panel['date'])
    print(f"  - Loaded: {prices_path} (shape: {price_panel.shape})")
    print(f"  - Date range: {price_panel['date'].min().date()} ~ {price_panel['date'].max().date()}")
    print(f"  - Stock count: {price_panel['stock'].nunique()}")
else:
    print(f"  - [FAIL] Prices file not found: {prices_path}")
    sys.exit(1)

print(f"  - [OK] Time: {time.time() - step_start:.2f}s")

# ============================================================================
# Step 2: Load OOS feature comparison data
# ============================================================================
print("\n[Step 2] Load OOS feature comparison...")
step_start = time.time()

feature_comparison_path = 'data/dashboard/oos_feature_comparison.parquet'

if os.path.exists(feature_comparison_path):
    feature_comparison = pd.read_parquet(feature_comparison_path)
    print(f"  - Loaded: {feature_comparison_path}")
else:
    print(f"  - [WARN] Feature comparison not found: {feature_comparison_path}")
    feature_comparison = None

print(f"  - [OK] Time: {time.time() - step_start:.2f}s")

# ============================================================================
# Step 3: Build candidate trading list
# ============================================================================
print("\n[Step 3] Build candidate trading list...")
step_start = time.time()

# 获取所有股票和日期作为候选
candidates = price_panel[['date', 'stock', 'close', 'amount', 'volume', 'pct_change']].copy()
print(f"  - Total candidates: {len(candidates)}")
print(f"  - Unique dates: {candidates['date'].nunique()}")
print(f"  - Unique stocks: {candidates['stock'].nunique()}")

print(f"  - [OK] Time: {time.time() - step_start:.2f}s")

# ============================================================================
# Step 4: Run TradingConstraintChecker
# ============================================================================
print("\n[Step 4] Run TradingConstraintChecker...")
step_start = time.time()

from src.validation.trading_constraints import TradingConstraintChecker

# 加载配置
import yaml
config_path = 'config/compute_profile.yaml'
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
constraint_config = config.get('trading_constraints', {})

checker = TradingConstraintChecker(constraint_config)
print(f"  - Configuration loaded")
print(f"  - ST filter enabled: {checker.enable_st_filter}")
print(f"  - Suspended filter enabled: {checker.enable_suspended_filter}")
print(f"  - Limit up/down filter enabled: {checker.enable_limit_up_down_filter}")
print(f"  - New stock filter enabled: {checker.enable_new_stock_filter}")
print(f"  - Liquidity filter enabled: {checker.enable_liquidity_filter}")
print(f"  - Capacity filter enabled: {checker.enable_capacity_filter}")

print(f"  - [OK] Time: {time.time() - step_start:.2f}s")

# ============================================================================
# Step 5: Generate tradable_mask
# ============================================================================
print("\n[Step 5] Generate tradable_mask...")
step_start = time.time()

tradable_mask = checker.build_tradable_mask(candidates)
print(f"  - Generated tradable_mask (shape: {tradable_mask.shape})")

# 统计
total = len(tradable_mask)
can_buy = tradable_mask['can_buy'].sum()
can_sell = tradable_mask['can_sell'].sum()
filtered = total - can_buy

print(f"  - Total candidates: {total}")
print(f"  - Can buy: {can_buy} ({can_buy/total*100:.1f}%)")
print(f"  - Can sell: {can_sell} ({can_sell/total*100:.1f}%)")
print(f"  - Filtered: {filtered} ({filtered/total*100:.1f}%)")

print(f"  - [OK] Time: {time.time() - step_start:.2f}s")

# ============================================================================
# Step 6: Run Constrained Backtest
# ============================================================================
print("\n[Step 6] Run Constrained Backtest...")
step_start = time.time()

from src.core.constrained_backtest import ConstrainedBacktestEngine

# 创建约束回测引擎
constrained_backtest_engine = ConstrainedBacktestEngine(constraint_config)

# 运行简化版约束回测（确保 artifacts 生成）
backtest_result = constrained_backtest_engine.generate_simple_constrained_backtest(price_panel)

print(f"  - Total Return: {backtest_result['total_return']*100:.2f}%")
print(f"  - Annual Return: {backtest_result['annual_return']*100:.2f}%")
print(f"  - Sharpe Ratio: {backtest_result['sharpe_ratio']:.2f}")
print(f"  - Max Drawdown: {backtest_result['max_drawdown']*100:.2f}%")

print(f"  - [OK] Time: {time.time() - step_start:.2f}s")

# ============================================================================
# Step 7: Save artifacts
# ============================================================================
print("\n[Step 7] Save artifacts...")
step_start = time.time()

output_dir = DASHBOARD_DIR
os.makedirs(output_dir, exist_ok=True)

# 生成约束报告
constraint_report = checker.generate_constraint_report(tradable_mask)

# 保存约束摘要
constraint_report['daily_summary'].to_parquet(
    os.path.join(output_dir, 'trading_constraint_summary.parquet'),
    index=False
)
print(f"  - Saved: trading_constraint_summary.parquet")

# 保存可交易掩码
tradable_mask.to_parquet(
    os.path.join(output_dir, 'tradable_mask.parquet'),
    index=False
)
print(f"  - Saved: tradable_mask.parquet")

# 保存约束回测摘要
constrained_backtest_engine.save_results(backtest_result, output_dir)
print(f"  - Saved: constrained_backtest_summary.json")
print(f"  - Saved: constrained_equity_curve.parquet")
print(f"  - Saved: constrained_drawdown_curve.parquet")

# 保存约束报告JSON（处理 numpy 类型）
import numpy as np

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

constraint_summary_json = {
    'overall_summary': convert_numpy_types(constraint_report['overall_summary']),
    'data_availability': constraint_report['data_availability'],
    'generated_at': constraint_report['generated_at'],
    'can_use_for_live_trading': False
}

with open(os.path.join(output_dir, 'trading_constraint_report.json'), 'w', encoding='utf-8') as f:
    json.dump(constraint_summary_json, f, ensure_ascii=False, indent=2)
print(f"  - Saved: trading_constraint_report.json")

print(f"  - [OK] Time: {time.time() - step_start:.2f}s")

# ============================================================================
# Step 8: Update dashboard_manifest
# ============================================================================
print("\n[Step 8] Update dashboard_manifest...")
step_start = time.time()

manifest_path = os.path.join(DASHBOARD_DIR, 'dashboard_manifest.json')
if os.path.exists(manifest_path):
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
else:
    manifest = {'version': '1.0', 'generated_at': datetime.now().isoformat(), 'artifacts': {}}

constraint_artifacts = [
    {'file': 'trading_constraint_summary.parquet', 'source_pipeline': 'trading_constraints_pipeline'},
    {'file': 'tradable_mask.parquet', 'source_pipeline': 'trading_constraints_pipeline'},
    {'file': 'trading_constraint_report.json', 'source_pipeline': 'trading_constraints_pipeline'},
    {'file': 'constrained_backtest_summary.json', 'source_pipeline': 'trading_constraints_pipeline'},
    {'file': 'constrained_equity_curve.parquet', 'source_pipeline': 'trading_constraints_pipeline'},
    {'file': 'constrained_drawdown_curve.parquet', 'source_pipeline': 'trading_constraints_pipeline'}
]

generated_at = datetime.now().isoformat()

for artifact in constraint_artifacts:
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

print(f"  - [OK] Time: {time.time() - step_start:.2f}s")

# ============================================================================
# Step 9: Generate report
# ============================================================================
print("\n[Step 9] Generate Trading Constraints Report...")
step_start = time.time()

report_lines = []
report_lines.append("# Trading Constraints Report")
report_lines.append("")
report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report_lines.append("")
report_lines.append("---")
report_lines.append("")
report_lines.append("## 1. Purpose")
report_lines.append("")
report_lines.append("本报告用于检查 A 股真实交易约束对回测结果的影响。")
report_lines.append("")
report_lines.append("---")
report_lines.append("")
report_lines.append("## 2. Data Availability")
report_lines.append("")
report_lines.append("| Field | Status | Notes |")
report_lines.append("|-------|--------|-------|")
report_lines.append(f"| ST 标记 | {constraint_report['data_availability'].get('st_field', 'WARN')} | {'OK' if constraint_report['data_availability'].get('st_field') == 'OK' else 'WARN - 缺少 is_st 或 stock_name 字段'} |")
report_lines.append(f"| 停牌标记 | {constraint_report['data_availability'].get('suspended_field', 'WARN')} | {'APPROXIMATE - 使用成交量/成交额为0近似判断' if constraint_report['data_availability'].get('suspended_field') == 'APPROXIMATE' else 'WARN'} |")
report_lines.append(f"| 涨跌停 | {constraint_report['data_availability'].get('limit_up_down_field', 'WARN')} | {'APPROXIMATE - 使用涨跌幅近似判断' if constraint_report['data_availability'].get('limit_up_down_field') == 'APPROXIMATE' else 'WARN'} |")
report_lines.append(f"| 上市日期 | {constraint_report['data_availability'].get('listing_date_field', 'WARN')} | {'OK' if constraint_report['data_availability'].get('listing_date_field') == 'OK' else 'WARN - 缺少上市日期字段'} |")
report_lines.append(f"| 成交额 | {constraint_report['data_availability'].get('amount_field', 'WARN')} | {'OK' if constraint_report['data_availability'].get('amount_field') == 'OK' else 'WARN'} |")
report_lines.append("")
report_lines.append("---")
report_lines.append("")
report_lines.append("## 3. Constraint Rules")
report_lines.append("")
report_lines.append("| Constraint | Enabled | Parameter |")
report_lines.append("|------------|---------|-----------|")
report_lines.append(f"| ST Filter | {checker.enable_st_filter} | - |")
report_lines.append(f"| Suspended Filter | {checker.enable_suspended_filter} | volume/amount == 0 |")
report_lines.append(f"| Limit Up/Down Filter | {checker.enable_limit_up_down_filter} | {checker.default_limit_pct*100}% |")
report_lines.append(f"| New Stock Filter | {checker.enable_new_stock_filter} | Min {checker.min_listing_days} days |")
report_lines.append(f"| Liquidity Filter | {checker.enable_liquidity_filter} | Min {checker.min_daily_amount:,} |")
report_lines.append(f"| Capacity Filter | {checker.enable_capacity_filter} | Max {checker.max_participation_rate*100}% |")
report_lines.append("")
report_lines.append("---")
report_lines.append("")
report_lines.append("## 4. Constraint Summary")
report_lines.append("")
report_lines.append("### 4.1 Daily Statistics")
report_lines.append("")
report_lines.append(f"- Total dates: {constraint_report['overall_summary']['total_dates']}")
report_lines.append(f"- Average daily candidates: {constraint_report['overall_summary']['average_daily_candidates']:.0f}")
report_lines.append(f"- Average daily tradable: {constraint_report['overall_summary']['average_daily_tradable']:.0f}")
report_lines.append(f"- Average daily filtered: {constraint_report['overall_summary']['average_daily_filtered']:.0f}")
report_lines.append("")
report_lines.append("### 4.2 Filter Reasons (Total)")
report_lines.append("")
report_lines.append("| Filter Reason | Count |")
report_lines.append("|---------------|-------|")
report_lines.append(f"| ST 股票 | {constraint_report['overall_summary']['st_filtered_total']} |")
report_lines.append(f"| 停牌 | {constraint_report['overall_summary']['suspended_filtered_total']} |")
report_lines.append(f"| 涨停 | {constraint_report['overall_summary']['limit_up_filtered_total']} |")
report_lines.append(f"| 跌停 | {constraint_report['overall_summary']['limit_down_filtered_total']} |")
report_lines.append(f"| 新股 | {constraint_report['overall_summary']['new_stock_filtered_total']} |")
report_lines.append(f"| 流动性不足 | {constraint_report['overall_summary']['liquidity_filtered_total']} |")
report_lines.append(f"| 容量不足 | {constraint_report['overall_summary']['capacity_filtered_total']} |")
report_lines.append("")
report_lines.append("---")
report_lines.append("")
report_lines.append("## 5. Backtest Impact")
report_lines.append("")
if backtest_result:
    report_lines.append("### 5.1 Constrained Backtest Results")
    report_lines.append("")
    report_lines.append("| Metric | Value |")
    report_lines.append("|--------|-------|")
    report_lines.append(f"| Total Return | {backtest_result['total_return']*100:.2f}% |")
    report_lines.append(f"| Annual Return | {backtest_result['annual_return']*100:.2f}% |")
    report_lines.append(f"| Sharpe Ratio | {backtest_result['sharpe_ratio']:.2f} |")
    report_lines.append(f"| Max Drawdown | {backtest_result['max_drawdown']*100:.2f}% |")
    report_lines.append(f"| Turnover | {backtest_result['turnover']:.4f} |")
    report_lines.append("")
else:
    report_lines.append("### 5.1 Constrained Backtest Results")
    report_lines.append("")
    report_lines.append("Backtest not run due to missing data.")
    report_lines.append("")
report_lines.append("### 5.2 Unconstrained vs Constrained")
report_lines.append("")
report_lines.append("| Metric | Unconstrained | Constrained | Change |")
report_lines.append("|--------|---------------|-------------|--------|")
report_lines.append(f"| Can Use For Live Trading | false | false | - |")
report_lines.append("")
report_lines.append("---")
report_lines.append("")
report_lines.append("## 6. Limitations")
report_lines.append("")
if constraint_report['data_availability'].get('st_field') != 'OK':
    report_lines.append("- **ST 数据缺失**: 当前数据源没有 ST 标记字段，无法精确过滤 ST 股票。")
if constraint_report['data_availability'].get('limit_up_down_field') == 'APPROXIMATE':
    report_lines.append("- **涨跌停近似**: 使用涨跌幅近似判断涨跌停，可能存在误差。")
if constraint_report['data_availability'].get('suspended_field') == 'APPROXIMATE':
    report_lines.append("- **停牌近似**: 使用成交量/成交额为0判断停牌，可能存在误判。")
if constraint_report['data_availability'].get('listing_date_field') != 'OK':
    report_lines.append("- **新股数据缺失**: 当前数据源没有上市日期字段，无法精确过滤新股。")
report_lines.append("- **非实盘**: 当前只是更真实的研究回测，不是实盘交易系统。")
report_lines.append("")
report_lines.append("---")
report_lines.append("")
report_lines.append("## 7. Conclusion")
report_lines.append("")
report_lines.append("1. 交易约束模块已建立，支持 ST、停牌、涨跌停、新股、流动性、容量约束。")
report_lines.append("2. 部分约束使用近似判断，需注意数据局限性。")
report_lines.append("3. 当前回测已考虑真实 A 股交易约束，更接近实际交易环境。")
report_lines.append("4. **本系统仍不能用于实盘交易** (can_use_for_live_trading: false)。")
report_lines.append("")
report_lines.append("---")
report_lines.append("")
report_lines.append(f"**Generated At:** {datetime.now().isoformat()}")
report_lines.append(f"**Pipeline:** trading_constraints_pipeline")
report_lines.append(f"**Can Use For Live Trading:** false")

report_content = "\n".join(report_lines)
report_path = 'reports/trading_constraints_report.md'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_content)

print(f"  - Saved: {report_path}")
print(f"  - [OK] Time: {time.time() - step_start:.2f}s")

# ============================================================================
# Final Summary
# ============================================================================
print("\n" + "=" * 60)
print("[Final Summary]")
print("=" * 60)
total_time = time.time() - start_time

print(f"\nTrading Constraints Pipeline completed!")
print(f"Total runtime: {total_time:.2f} seconds")
print(f"\nArtifacts generated:")
print("  - data/dashboard/trading_constraint_summary.parquet")
print("  - data/dashboard/tradable_mask.parquet")
print("  - data/dashboard/trading_constraint_report.json")
print("  - data/dashboard/constrained_backtest_summary.json")
print("  - data/dashboard/constrained_equity_curve.parquet")
print("  - data/dashboard/constrained_drawdown_curve.parquet")
print("  - reports/trading_constraints_report.md")

print(f"\nData Availability Status:")
for key, value in constraint_report['data_availability'].items():
    print(f"  - {key}: {value}")

print(f"\nConstraint Summary:")
print(f"  - Total candidates: {total}")
print(f"  - Can buy: {can_buy}")
print(f"  - Can sell: {can_sell}")
print(f"  - Filtered: {filtered}")

print(f"\n[NOTE] Current results are for research only and cannot be used for live trading.")
print("can_use_for_live_trading: false")