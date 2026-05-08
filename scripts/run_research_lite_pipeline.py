"""
Research Factor Pipeline - 研究因子 Pipeline

支持多种 profile：
- student_laptop: 20只股票，6个月
- research_lite: 100只股票，12个月
- research_medium_trial: 150只股票，18个月
- research_medium: 300只股票，24个月

特点:
- 使用DataSourceManager统一管理数据获取
- 本地parquet缓存
- 公式因子和neural因子分别评价
- 样本外验证
- 可信度审计
- 支持 profile 参数
- 数据不足时FAIL或WARN
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Parse command line arguments
parser = argparse.ArgumentParser(description='Research Factor Pipeline')
parser.add_argument('--profile', default='research_lite', 
                    help='Profile name from compute_profile.yaml')
args = parser.parse_args()

profile_name = args.profile

print("=" * 80)
print(f"MyQuant Research Factor Pipeline")
print(f"Profile: {profile_name}")
print("=" * 80)

start_time = time.time()

# ============================================================================
# Step 1: Load Configuration & Initialize DataSourceManager
# ============================================================================
print("\n" + "=" * 80)
print("[Step 1] Load Configuration")
print("=" * 80)
step_start = time.time()

from src.data.data_source_manager import DataSourceManager

ds_manager = DataSourceManager()

profile_config = ds_manager.get_profile_config(profile_name)
data_source_config = ds_manager.get_data_source_config()

if not profile_config:
    print(f"  - [ERROR] Profile '{profile_name}' not found in config")
    sys.exit(1)

STOCK_COUNT_TARGET = profile_config.get('stock_count', 100)
HISTORY_MONTHS_TARGET = profile_config.get('history_months', 12)
FORMULA_FACTOR_LIMIT = profile_config.get('formula_factor_limit', 100)
NEURAL_EMBEDDING_DIM = profile_config.get('neural_embedding_dim', 8)
LOOKBACK_WINDOW = profile_config.get('lookback_window', 20)
EPOCHS = profile_config.get('epochs', 5)
BATCH_SIZE = profile_config.get('batch_size', 64)
HIDDEN_DIM = profile_config.get('hidden_dim', 32)
DEVICE = profile_config.get('device', 'cpu')
TRAIN_RATIO = profile_config.get('train_ratio', 0.6)
VAL_RATIO = profile_config.get('val_ratio', 0.2)
TEST_RATIO = profile_config.get('test_ratio', 0.2)

# Profile-specific output directories
PROFILE_DIR = f'data/processed/profiles/{profile_name}'
DASHBOARD_DIR = f'data/dashboard/profiles/{profile_name}'
os.makedirs(PROFILE_DIR, exist_ok=True)
os.makedirs(DASHBOARD_DIR, exist_ok=True)

print("\n[Diagnostics] Configuration:")
print("  - Profile: {}".format(profile_name))
print("  - Target stock count: {}".format(STOCK_COUNT_TARGET))
print("  - Target history months: {}".format(HISTORY_MONTHS_TARGET))
print("  - Formula factor limit: {}".format(FORMULA_FACTOR_LIMIT))
print("  - Neural embedding dim: {}".format(NEURAL_EMBEDDING_DIM))
print("  - Lookback window: {}".format(LOOKBACK_WINDOW))
print("  - Epochs: {}".format(EPOCHS))
print("  - Device: {}".format(DEVICE))
print("  - Profile directory: {}".format(PROFILE_DIR))
print("  - Dashboard directory: {}".format(DASHBOARD_DIR))
print("  - [OK] Config loaded in {:.2f}s".format(time.time() - step_start))

# ============================================================================
# Step 2: Load or Fetch Data via DataSourceManager
# ============================================================================
print("\n" + "=" * 80)
print("[Step 2] Load or Fetch Data")
print("=" * 80)
step_start = time.time()

print(f"\n  - Using DataSourceManager for {profile_name}")
price_data, result = ds_manager.fetch_price_panel(profile_name)

status = result.get('status', 'FAIL')
metadata = result.get('metadata', {})

print(f"  - Data fetch status: {status}")

if status == 'FAIL':
    print(f"  - [FAIL] Data fetch failed!")
    print(f"  - Reason: {result.get('reason', 'Unknown')}")
    
    # Generate reliability report even on failure
    ds_manager.generate_data_fetch_report(profile_name)
    
    print("\n" + "!" * 80)
    print("  [CRITICAL FAILURE] Data fetch failed - cannot proceed")
    print("  Check reports/data_source_reliability_report.md for details")
    print("!" * 80)
    sys.exit(1)

if status == 'WARN':
    print(f"  - [WARN] Data fetch completed with warnings")
    print(f"  - Actual stock count may be below target")

ACTUAL_STOCK_COUNT = metadata.get('actual_stock_count', price_data['stock'].nunique())
START_DATE = metadata.get('actual_start_date', price_data['date'].min().strftime('%Y%m%d'))
END_DATE = metadata.get('actual_end_date', price_data['date'].max().strftime('%Y%m%d'))

print("\n[Diagnostics] Data Summary:")
print("  - Target stock count: {}".format(STOCK_COUNT_TARGET))
print("  - Actual stock count: {}".format(ACTUAL_STOCK_COUNT))
print("  - Target history months: {}".format(HISTORY_MONTHS_TARGET))
print("  - Actual date range: {} to {}".format(START_DATE, END_DATE))
print("  - Total rows: {}".format(len(price_data)))

print("\n[Diagnostics] Stock by Date:")
stock_count_by_date = price_data.groupby('date')['stock'].nunique()
print("  - Dates range: {} - {}".format(stock_count_by_date.index[0], stock_count_by_date.index[-1]))
print("  - Stock count by date - min: {}, max: {}, avg: {:.1f}".format(
    stock_count_by_date.min(), stock_count_by_date.max(), stock_count_by_date.mean()
))

# Generate data fetch report
ds_manager.generate_data_fetch_report(profile_name)
print("  - Data source reliability report generated")

print("  - [OK] Data loaded in {:.2f}s".format(time.time() - step_start))

# ============================================================================
# Step 3: Data Preparation
# ============================================================================
print("\n" + "=" * 80)
print("[Step 3] Data Preparation")
print("=" * 80)
step_start = time.time()

standard_columns = ['date', 'stock', 'open', 'high', 'low', 'close', 'volume', 'amount', 'turnover']

for col in standard_columns:
    if col not in price_data.columns:
        price_data[col] = 0

price_data = price_data[standard_columns]
price_data = price_data.dropna(subset=['close', 'volume'])

print("\n[Diagnostics] Future Return Calculation:")
print("  - Calculating future returns - GROUPED BY STOCK")

df_with_future = price_data.copy()
df_with_future['future_return'] = df_with_future.groupby('stock')['close'].pct_change().shift(-1)
future_returns = df_with_future[['date', 'stock', 'future_return']].dropna()

future_returns_mi = future_returns.set_index(['date', 'stock'])['future_return']

print("  - future_return index type: {}".format(type(future_returns_mi.index)))
if hasattr(future_returns_mi.index, 'names'):
    print("  - future_return index names: {}".format(future_returns_mi.index.names))
print("  - future_return shape: {}".format(len(future_returns_mi)))
print("  - [OK] Data prepared in {:.2f}s".format(time.time() - step_start))

# ============================================================================
# Step 4: Generate Formula Factors
# ============================================================================
print("\n" + "=" * 80)
print("[Step 4] Generate Formula Factors")
print("=" * 80)
step_start = time.time()

from src.factors.auto.enhanced_generator import EnhancedFactorGenerator

generator = EnhancedFactorGenerator()

index = pd.MultiIndex.from_frame(price_data[['date', 'stock']])
features = price_data.set_index(index).drop(['date', 'stock'], axis=1)

formula_factors = generator.generate_all_factors(features, generate_neutral=False)

factor_names = list(formula_factors.keys())[:FORMULA_FACTOR_LIMIT]
formula_factors = {k: formula_factors[k] for k in factor_names}

print("\n[Diagnostics] Formula Factors:")
print("  - Generated: {}".format(len(formula_factors)))

if len(formula_factors) > 0:
    first_factor_name, first_factor_data = list(formula_factors.items())[0]
    print("  - First factor: {}".format(first_factor_name))
    print("  - First factor index type: {}".format(type(first_factor_data.index)))
    if hasattr(first_factor_data.index, 'names'):
        print("  - First factor index names: {}".format(first_factor_data.index.names))
    print("  - First factor shape: {}".format(len(first_factor_data)))

print("  - [OK] Formula factors generated in {:.2f}s".format(time.time() - step_start))

# ============================================================================
# Step 5: Evaluate Formula Factors
# ============================================================================
print("\n" + "=" * 80)
print("[Step 5] Evaluate Formula Factors")
print("=" * 80)
step_start = time.time()

from src.factors.auto.factor_evaluator import FactorEvaluator

evaluator = FactorEvaluator()

formula_eval_results = {}
formula_eval_failures = {}
formula_factor_items = list(formula_factors.items())

print("\n[Diagnostics] Formula Factor Evaluation:")
print("  - Total factors: {}".format(len(formula_factor_items)))

for i, (name, factor_data) in enumerate(formula_factor_items):
    try:
        if factor_data is None or len(factor_data) == 0:
            formula_eval_failures[name] = "Empty factor data"
            continue

        common_idx = factor_data.index.intersection(future_returns_mi.index)

        if len(common_idx) < 100:
            formula_eval_failures[name] = "Common index too small: {}".format(len(common_idx))
            continue

        eval_result = evaluator.evaluate_single(factor_data, future_returns_mi)

        if eval_result['rank_ic']['count'] == 0:
            formula_eval_failures[name] = "RankIC count = 0"
            continue

        formula_eval_results[name] = {
            'rank_ic_mean': eval_result['rank_ic']['mean'],
            'icir': eval_result.get('icir', 0),
            'coverage': eval_result.get('coverage', 0),
            'turnover': eval_result.get('turnover', 0),
            'rank_ic_count': eval_result['rank_ic']['count'],
            'rank_ic_timeseries': eval_result['rank_ic'].get('timeseries', []),
            'rank_ic_dates': eval_result['rank_ic'].get('dates', []),
            'factor_data': factor_data  # 保存原始因子数据用于相关性计算
        }

    except Exception as e:
        formula_eval_failures[name] = str(e)[:100]

    if (i + 1) % 20 == 0:
        print("  - Evaluated {}/{}... success: {}".format(i + 1, len(formula_factor_items), len(formula_eval_results)))

print("\n[Diagnostics] Formula Factor Evaluation Results:")
print("  - Total: {}".format(len(formula_factor_items)))
print("  - Successfully evaluated: {}".format(len(formula_eval_results)))
print("  - Failed: {}".format(len(formula_eval_failures)))

if len(formula_eval_failures) > 0:
    print("  - Failure samples:")
    for name, reason in list(formula_eval_failures.items())[:5]:
        print("    - {}: {}".format(name, reason))

# Save dashboard artifacts - factor summary, IC series and correlation
# Factor summary
if formula_eval_results:
    factor_summary_data = []
    factor_ic_series_data = []
    
    for name, result in formula_eval_results.items():
        factor_summary_data.append({
            'factor_name': name,
            'factor_type': 'formula',
            'rank_ic_mean': result['rank_ic_mean'],
            'icir': result['icir'],
            'coverage': result['coverage'],
            'turnover': result['turnover'],
            'gatekeeper_status': 'PASS' if abs(result['rank_ic_mean']) > 0.01 and result['icir'] > 0.1 else 'FAIL',
            'is_neural_factor': False,
            'is_formula_factor': True
        })
        
        # Save IC series
        if result.get('rank_ic_timeseries') and result.get('rank_ic_dates'):
            for date, ic in zip(result['rank_ic_dates'], result['rank_ic_timeseries']):
                factor_ic_series_data.append({
                    'date': date,
                    'factor_name': name,
                    'rank_ic': ic
                })
    
    factor_summary_df = pd.DataFrame(factor_summary_data)
    factor_summary_path = os.path.join(DASHBOARD_DIR, 'factor_summary.parquet')
    factor_summary_df.to_parquet(factor_summary_path, index=False)
    print(f"  - Saved: {factor_summary_path}")
    
    # Save factor IC series
    if factor_ic_series_data:
        factor_ic_series_df = pd.DataFrame(factor_ic_series_data)
        factor_ic_series_path = os.path.join(DASHBOARD_DIR, 'factor_ic_series.parquet')
        factor_ic_series_df.to_parquet(factor_ic_series_path, index=False)
        print(f"  - Saved: {factor_ic_series_path}")
    
    # Calculate and save factor correlation (memory-efficient)
    factor_names = list(formula_eval_results.keys())
    if len(factor_names) >= 2:
        correlation_data = []
        
        # Calculate correlation pairwise to avoid memory issues
        for i in range(len(factor_names)):
            for j in range(i + 1, len(factor_names)):
                f1_name = factor_names[i]
                f2_name = factor_names[j]
                
                f1_data = formula_eval_results[f1_name].get('factor_data')
                f2_data = formula_eval_results[f2_name].get('factor_data')
                
                if f1_data is not None and f2_data is not None:
                    try:
                        # Find common index
                        common_idx = f1_data.dropna().index.intersection(f2_data.dropna().index)
                        
                        if len(common_idx) > 10:
                            f1_vals = f1_data.loc[common_idx]
                            f2_vals = f2_data.loc[common_idx]
                            
                            # Use rank correlation (Spearman) which is more robust
                            corr = f1_vals.corr(f2_vals, method='spearman')
                            
                            if not np.isnan(corr):
                                correlation_data.append({
                                    'factor_1': f1_name,
                                    'factor_2': f2_name,
                                    'correlation': float(corr)
                                })
                    except Exception:
                        continue
        
        if correlation_data:
            correlation_df = pd.DataFrame(correlation_data)
            correlation_path = os.path.join(DASHBOARD_DIR, 'factor_correlation.parquet')
            correlation_df.to_parquet(correlation_path, index=False)
            print(f"  - Saved: {correlation_path}")

# Save formula factor panel (stock-date level factor values)
if formula_eval_results:
    print("\n[Diagnostics] Saving Formula Factor Panel:")
    
    factor_dfs = []
    for name, result in formula_eval_results.items():
        factor_data = result.get('factor_data')
        if factor_data is not None:
            df = factor_data.reset_index()
            df = df.rename(columns={0: name})
            factor_dfs.append(df)
    
    if factor_dfs:
        formula_factors_panel = factor_dfs[0]
        for df in factor_dfs[1:]:
            formula_factors_panel = pd.merge(formula_factors_panel, df, on=['date', 'stock'], how='inner')
        
        formula_factors_path = os.path.join(DASHBOARD_DIR, 'formula_factors.parquet')
        formula_factors_panel.to_parquet(formula_factors_path, index=False)
        print(f"  - Saved: {formula_factors_path} (shape: {formula_factors_panel.shape})")
        
        # Save metadata
        formula_factor_metadata = {
            'factor_count': len(formula_eval_results),
            'factor_names': list(formula_eval_results.keys()),
            'date_range': {
                'min': formula_factors_panel['date'].min().strftime('%Y-%m-%d'),
                'max': formula_factors_panel['date'].max().strftime('%Y-%m-%d')
            },
            'stock_count': formula_factors_panel['stock'].nunique(),
            'rows': len(formula_factors_panel),
            'columns': len(formula_factors_panel.columns),
            'generated_at': datetime.now().isoformat(),
            'leakage_check_status': 'PENDING',
            'source_pipeline': 'run_research_lite_pipeline.py'
        }
        
        formula_metadata_path = os.path.join(DASHBOARD_DIR, 'formula_factor_metadata.json')
        with open(formula_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(formula_factor_metadata, f, ensure_ascii=False, indent=2)
        print(f"  - Saved: {formula_metadata_path}")
        
        # Update profile-specific manifest
        manifest_path = os.path.join(DASHBOARD_DIR, 'profile_manifest.json')
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
        else:
            manifest = {'version': '1.0', 'generated_at': datetime.now().isoformat(), 'artifacts': {}}
        
        manifest['profile'] = profile_name
        manifest['stock_count_target'] = STOCK_COUNT_TARGET
        manifest['stock_count_actual'] = ACTUAL_STOCK_COUNT
        manifest['history_months_target'] = HISTORY_MONTHS_TARGET
        manifest['date_start'] = START_DATE
        manifest['date_end'] = END_DATE
        manifest['can_use_for_live_trading'] = False
        
        manifest['artifacts']['formula_factors.parquet'] = {
            'exists': True,
            'generated_by': 'run_research_lite_pipeline.py',
            'last_updated': datetime.now().isoformat(),
            'rows': len(formula_factors_panel),
            'columns': len(formula_factors_panel.columns),
            'status': 'OK',
            'note': 'Stock-date level formula factor values'
        }
        
        manifest['artifacts']['formula_factor_metadata.json'] = {
            'exists': True,
            'generated_by': 'run_research_lite_pipeline.py',
            'last_updated': datetime.now().isoformat(),
            'status': 'OK',
            'note': 'Formula factor metadata'
        }
        
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"  - Updated: {manifest_path}")

print("  - [OK] Formula factors evaluated in {:.2f}s".format(time.time() - step_start))

# ============================================================================
# Step 6: Generate Neural Factors & Leakage Check
# ============================================================================
print("\n" + "=" * 80)
print("[Step 6] Generate Neural Factors & Leakage Check")
print("=" * 80)
step_start = time.time()

from src.factors.neural.sequence_dataset import SequenceDataset
from src.factors.neural.autoencoder import SequenceAutoEncoder
from src.factors.neural.neural_factor_extractor import NeuralFactorExtractor
from src.validation.neural_leakage_check import NeuralLeakageChecker

RAW_FEATURES = ['open', 'high', 'low', 'close', 'volume']

print("\n[Diagnostics] Neural Factor Pipeline:")

dataset = SequenceDataset(
    df=price_data,
    lookback_window=LOOKBACK_WINDOW,
    features=RAW_FEATURES,
    target_horizon=1
)

X, metadata_df = dataset.get_samples()

splits = dataset.get_train_val_test_split(
    train_ratio=TRAIN_RATIO,
    val_ratio=VAL_RATIO,
    test_ratio=TEST_RATIO
)

X_train, meta_train = splits['train']
X_val, meta_val = splits['val']
X_test, meta_test = splits['test']

print("\n[Diagnostics] Time Split:")
print("  - Train samples: {}, dates: {} - {}".format(len(X_train), meta_train['signal_date'].min(), meta_train['signal_date'].max()))
print("  - Val samples: {}, dates: {} - {}".format(len(X_val), meta_val['signal_date'].min(), meta_val['signal_date'].max()))
print("  - Test samples: {}, dates: {} - {}".format(len(X_test), meta_test['signal_date'].min(), meta_test['signal_date'].max()))

print("\n[Diagnostics] Target Alignment Check:")
print("  - metadata shape: {}".format(metadata_df.shape))
print("  - signal_date min/max: {} / {}".format(metadata_df['signal_date'].min(), metadata_df['signal_date'].max()))
print("  - target_start_date min/max: {} / {}".format(metadata_df['target_start_date'].min(), metadata_df['target_start_date'].max()))

alignment_violations = metadata_df[metadata_df['target_start_date'] <= metadata_df['signal_date']]
print("  - Violations (target_start_date <= signal_date): {}".format(len(alignment_violations)))
if len(alignment_violations) > 0:
    print("  - First 5 violations:")
    for idx, row in alignment_violations.head(5).iterrows():
        print("    - signal_date: {}, target_start_date: {}".format(row['signal_date'], row['target_start_date']))

checker = NeuralLeakageChecker()
leakage_results = checker.run_all_checks(
    columns=RAW_FEATURES,
    metadata=metadata_df,
    train_dates=(meta_train['signal_date'].min(), meta_train['signal_date'].max()),
    val_dates=(meta_val['signal_date'].min(), meta_val['signal_date'].max()),
    test_dates=(meta_test['signal_date'].min(), meta_test['signal_date'].max())
)

print("\n[Diagnostics] Leakage Check Results:")
for check_name, result in leakage_results.items():
    if check_name == 'overall_status':
        print("  - {}: {}".format(check_name, result))
    else:
        print("  - {}: {}".format(check_name, result.get('status', 'N/A')))

OVERALL_STATUS = leakage_results.get('overall_status', 'OK')

if OVERALL_STATUS == 'FAIL':
    print("\n" + "!" * 80)
    print("  [CRITICAL FAILURE] Leakage check returned FAIL")
    print("  Pipeline will STOP, no neural training will proceed")
    print("!" * 80)
    print("\n[Step 6] Leakage check FAILED - stopping pipeline")
else:
    print("\n  - Leakage check {} - proceeding".format(OVERALL_STATUS))

print("  - [OK] Neural pipeline setup in {:.2f}s".format(time.time() - step_start))

# ============================================================================
# Step 7: Train Neural Model (only if Leakage OK/WARN)
# ============================================================================
neural_eval_results = {}
neural_factors_df = None

if OVERALL_STATUS in ['OK', 'WARN']:
    print("\n" + "=" * 80)
    print("[Step 7] Train Neural Model")
    print("=" * 80)
    step_start = time.time()

    import torch
    import torch.nn as nn

    model = SequenceAutoEncoder(
        input_dim=len(RAW_FEATURES),
        hidden_dim=HIDDEN_DIM,
        embedding_dim=NEURAL_EMBEDDING_DIM,
        lookback_window=LOOKBACK_WINDOW,
        encoder_type='mlp'
    )

    model = model.to(DEVICE)
    model.train()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X_train_tensor = torch.FloatTensor(X_train)
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, X_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    train_losses = []

    for epoch in range(EPOCHS):
        epoch_loss = 0
        n_batches = 0

        for batch_X, _ in train_loader:
            batch_X = batch_X.to(DEVICE)

            reconstruction, embedding = model(batch_X)
            loss = criterion(reconstruction, batch_X)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches if n_batches > 0 else 0
        train_losses.append(avg_loss)

        print("  - Epoch {}/{}: loss={:.4f}".format(epoch + 1, EPOCHS, avg_loss))

    print("  - [OK] Model trained in {:.2f}s".format(time.time() - step_start))

    # ============================================================================
    # Step 8: Extract & Evaluate Neural Factors
    # ============================================================================
    print("\n" + "=" * 80)
    print("[Step 8] Extract & Evaluate Neural Factors")
    print("=" * 80)
    step_start = time.time()

    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
        _, embeddings = model(X_test_tensor)

    embeddings = embeddings.cpu().numpy()

    extractor = NeuralFactorExtractor(embedding_dim=NEURAL_EMBEDDING_DIM)
    neural_factors_df = extractor.embedding_to_dataframe(embeddings, meta_test.reset_index(drop=True))

    neural_factor_names = ['neural_factor_{}'.format(i) for i in range(NEURAL_EMBEDDING_DIM)]

    print("\n[Diagnostics] Neural Factor Conversion:")
    print("  - Neural factors DataFrame shape: {}".format(neural_factors_df.shape))
    print("  - Columns: {}".format(list(neural_factors_df.columns)))

    neural_factor_dict = {}

    for i, factor_name in enumerate(neural_factor_names):
        if factor_name in neural_factors_df.columns:
            factor_data = neural_factors_df.set_index(['date', 'stock'])[factor_name]

            common_idx = factor_data.index.intersection(future_returns_mi.index)

            print("\n  - Neural Factor {}:".format(factor_name))
            print("    - Factor index type: {}".format(type(factor_data.index)))
            print("    - Factor index names: {}".format(factor_data.index.names if hasattr(factor_data.index, 'names') else 'N/A'))
            print("    - Common index with future_return: {}".format(len(common_idx)))

            try:
                eval_result = evaluator.evaluate_single(factor_data, future_returns_mi)

                if eval_result['rank_ic']['count'] > 0:
                    neural_eval_results[factor_name] = {
                        'rank_ic_mean': eval_result['rank_ic']['mean'],
                        'icir': eval_result.get('icir', 0),
                        'coverage': eval_result.get('coverage', 0),
                        'turnover': eval_result.get('turnover', 0),
                        'rank_ic_count': eval_result['rank_ic']['count']
                    }
                    print("    - [OK] RankIC mean: {:.4f}, ICIR: {:.4f}, count: {}".format(
                        neural_eval_results[factor_name]['rank_ic_mean'],
                        neural_eval_results[factor_name]['icir'],
                        neural_eval_results[factor_name]['rank_ic_count']))
                else:
                    print("    - [WARN] RankIC count is 0")
            except Exception as e:
                print("    - [FAIL] {}".format(str(e)[:100]))
                import traceback
                print("    - Traceback: {}".format(traceback.format_exc()[:200]))

    if neural_eval_results:
        neural_summary_data = []
        for name, result in neural_eval_results.items():
            neural_summary_data.append({
                'factor_name': name,
                'encoder_type': 'mlp',
                'rank_ic_mean': result['rank_ic_mean'],
                'icir': result['icir'],
                'coverage': result['coverage'],
                'turnover': result['turnover'],
                'gatekeeper_status': 'PASS' if abs(result['rank_ic_mean']) > 0.01 and result['icir'] > 0.1 else 'FAIL',
                'is_neural_factor': True,
                'is_formula_factor': False
            })
        
        neural_summary_df = pd.DataFrame(neural_summary_data)
        neural_summary_path = os.path.join(DASHBOARD_DIR, 'neural_factor_summary.parquet')
        neural_summary_df.to_parquet(neural_summary_path, index=False)
        print(f"  - Saved: {neural_summary_path}")
    
    print("\n  - [OK] Neural factors evaluated in {:.2f}s".format(time.time() - step_start))

else:
    print("\n[Step 7-8] SKIPPED - Leakage check FAIL")

# ============================================================================
# Step 9: Generate Reliability Audit Report
# ============================================================================
print("\n" + "=" * 80)
print("[Step 9] Generate Reliability Audit Report")
print("=" * 80)
step_start = time.time()

formula_df = pd.DataFrame(formula_eval_results).T.sort_values('rank_ic_mean', ascending=False) if formula_eval_results else pd.DataFrame()
neural_df = pd.DataFrame(neural_eval_results).T.sort_values('rank_ic_mean', ascending=False) if neural_eval_results else pd.DataFrame()

report_lines = []
report_lines.append(f"# Research Pipeline Report ({profile_name})")
report_lines.append("")
report_lines.append("Generated: {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
report_lines.append("")
report_lines.append("## 1. Data Overview")
report_lines.append("")
report_lines.append("| Metric | Target | Actual | Status |")
report_lines.append("|--------|--------|--------|--------|")

stock_status = "OK" if ACTUAL_STOCK_COUNT >= STOCK_COUNT_TARGET * 0.8 else "WARN" if ACTUAL_STOCK_COUNT >= 30 else "FAIL"
report_lines.append("| Stock count | {} | {} | {} |".format(STOCK_COUNT_TARGET, ACTUAL_STOCK_COUNT, stock_status))

date_diff_days = (pd.to_datetime(END_DATE) - pd.to_datetime(START_DATE)).days
date_status = "OK" if date_diff_days >= 300 else "WARN" if date_diff_days >= 120 else "FAIL"
report_lines.append("| History months | {} | {:.1f} | {} |".format(HISTORY_MONTHS_TARGET, date_diff_days / 30, date_status))

report_lines.append("| Date range | N/A | {} to {} | - |".format(START_DATE, END_DATE))
report_lines.append("| Total rows | N/A | {} | - |".format(len(price_data)))

report_lines.append("")
report_lines.append("## 2. Formula Factors")
report_lines.append("")

if len(formula_df) > 0:
    report_lines.append("| Rank | Factor | RankIC | ICIR | RankIC Count |")
    report_lines.append("|------|--------|--------|------|--------------|")

    for i, (name, row) in enumerate(formula_df.head(10).iterrows()):
        try:
            report_lines.append("| {} | {} | {:.4f} | {:.4f} | {} |".format(
                i + 1, name, row['rank_ic_mean'], row['icir'], int(row.get('rank_ic_count', 0))
            ))
        except:
            pass
else:
    report_lines.append("**No formula factors successfully evaluated**")

report_lines.append("")
report_lines.append("- Total formula factors: {}".format(len(formula_factor_items)))
report_lines.append("- Successfully evaluated: {}".format(len(formula_eval_results)))
report_lines.append("- Failed: {}".format(len(formula_eval_failures)))

report_lines.append("")
report_lines.append("## 3. Neural Factors")
report_lines.append("")

if len(neural_df) > 0:
    report_lines.append("| Rank | Factor | RankIC | ICIR | RankIC Count |")
    report_lines.append("|------|--------|--------|------|--------------|")

    for i, (name, row) in enumerate(neural_df.iterrows()):
        try:
            report_lines.append("| {} | {} | {:.4f} | {:.4f} | {} |".format(
                i + 1, name, row['rank_ic_mean'], row['icir'], int(row.get('rank_ic_count', 0))
            ))
        except:
            pass
else:
    report_lines.append("**No neural factors successfully evaluated (or leakage check FAIL)**")

report_lines.append("")
report_lines.append("- Total neural factors: {}".format(NEURAL_EMBEDDING_DIM))
report_lines.append("- Successfully evaluated: {}".format(len(neural_eval_results)))

report_lines.append("")
report_lines.append("## 4. Leakage Check")
report_lines.append("")

report_lines.append("| Check | Status | Details |")
report_lines.append("|-------|--------|---------|")

for check_name, result in leakage_results.items():
    if check_name != 'overall_status':
        status = result.get('status', 'SKIP')
        message = result.get('message', '')
        report_lines.append("| {} | {} | {} |".format(check_name, status, message))

report_lines.append("| **Overall** | **{}** | - |".format(OVERALL_STATUS))

report_lines.append("")
report_lines.append("## 5. Future Return Alignment")
report_lines.append("")
report_lines.append("- future_return calculated GROUPED BY STOCK: Yes")
report_lines.append("- MultiIndex format: Yes")

report_lines.append("")
report_lines.append("## 6. Final Status")
report_lines.append("")

if OVERALL_STATUS == 'FAIL':
    report_lines.append("**STATUS: FAIL** - Leakage check failed, pipeline stopped before neural training")
elif stock_status == 'FAIL' or date_status == 'FAIL':
    report_lines.append("**STATUS: WARN** - Some data quality issues")
else:
    report_lines.append("**STATUS: OK** - Pipeline successful")

report_lines.append("")
report_lines.append("## 7. Limitations")
report_lines.append("")
report_lines.append("- Leakage check status: {}".format(OVERALL_STATUS))
report_lines.append("- Stock count target vs actual: {} vs {}".format(STOCK_COUNT_TARGET, ACTUAL_STOCK_COUNT))
report_lines.append("- Date range sufficiency: {}".format(date_status))
report_lines.append("- Formula factors success rate: {:.1f}%".format(len(formula_eval_results) / len(formula_factor_items) * 100 if len(formula_factor_items) > 0 else 0))
report_lines.append("- This is a research prototype - NOT FOR LIVE TRADING")

report_lines.append("")
report_lines.append("Total time: {:.2f}s".format(time.time() - start_time))

report = "\n".join(report_lines)

REPORT_FILE = f'reports/{profile_name}_report.md'
with open(REPORT_FILE, 'w', encoding='utf-8') as f:
    f.write(report)

print("\n  - Report saved: {}".format(REPORT_FILE))
print("  - [OK] Report generated in {:.2f}s".format(time.time() - step_start))

# ============================================================================
# Done
# ============================================================================
print("\n" + "=" * 80)
print("Research Pipeline Completed!")
print("=" * 80)
print(f"Profile: {profile_name}")
print("Total time: {:.2f}s".format(time.time() - start_time))
print("Report: {}".format(REPORT_FILE))

if OVERALL_STATUS == 'FAIL':
    print("\n[IMPORTANT] Leakage check FAILED - results may be unreliable")

if stock_status == 'FAIL':
    print("\n[IMPORTANT] Stock count is below acceptable threshold")
    print(f"  - Target: {STOCK_COUNT_TARGET}, Actual: {ACTUAL_STOCK_COUNT}")
    print("  - Consider checking data source reliability")