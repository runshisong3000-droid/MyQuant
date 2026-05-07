"""
Research Lite Pipeline - 研究轻量模式 (修复版)

从 student_laptop 模式升级到小研究样本模式。

目标:
- 100只股票
- 1年历史数据
- 100个公式因子
- 8个neural factors

特点:
- 使用AkShare真实数据
- 本地parquet缓存
- 公式因子和neural因子分别评价
- 样本外验证
- 可信度审计
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

print("=" * 80)
print("MyQuant Research Lite Pipeline (Reliability Fix Version)")
print("=" * 80)

start_time = time.time()

# ============================================================================
# Step 1: Load Configuration
# ============================================================================
print("\n" + "=" * 80)
print("[Step 1] Load Configuration")
print("=" * 80)
step_start = time.time()

config_path = 'config/compute_profile.yaml'

if os.path.exists(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config_all = yaml.safe_load(f)

    config = config_all['profiles']['research_lite']
    print("  - Profile: research_lite")
    print("  - Config loaded from: {}".format(config_path))
else:
    print("  - [WARN] Config file not found, using defaults")
    config = {
        'stock_count': 100,
        'history_months': 12,
        'formula_factor_limit': 100,
        'neural_embedding_dim': 8,
        'lookback_window': 20,
        'epochs': 5,
        'batch_size': 64,
        'hidden_dim': 32,
        'device': 'cpu',
        'use_cache': True,
        'cache_dir': 'data/cache',
        'train_ratio': 0.6,
        'val_ratio': 0.2,
        'test_ratio': 0.2
    }

STOCK_COUNT_TARGET = config['stock_count']
HISTORY_MONTHS_TARGET = config['history_months']
FORMULA_FACTOR_LIMIT = config['formula_factor_limit']
NEURAL_EMBEDDING_DIM = config['neural_embedding_dim']
LOOKBACK_WINDOW = config['lookback_window']
EPOCHS = config['epochs']
BATCH_SIZE = config['batch_size']
HIDDEN_DIM = config['hidden_dim']
DEVICE = config['device']
USE_CACHE = config.get('use_cache', True)
CACHE_DIR = config.get('cache_dir', 'data/cache')
TRAIN_RATIO = config['train_ratio']
VAL_RATIO = config['val_ratio']
TEST_RATIO = config['test_ratio']

print("\n[Diagnostics] Configuration:")
print("  - Target stock count: {}".format(STOCK_COUNT_TARGET))
print("  - Target history months: {}".format(HISTORY_MONTHS_TARGET))
print("  - Formula factor limit: {}".format(FORMULA_FACTOR_LIMIT))
print("  - Neural embedding dim: {}".format(NEURAL_EMBEDDING_DIM))
print("  - Lookback window: {}".format(LOOKBACK_WINDOW))
print("  - Epochs: {}".format(EPOCHS))
print("  - Device: {}".format(DEVICE))
print("  - Use cache: {}".format(USE_CACHE))
print("  - [OK] Config loaded in {:.2f}s".format(time.time() - step_start))

# ============================================================================
# Step 2: Load or Fetch Data
# ============================================================================
print("\n" + "=" * 80)
print("[Step 2] Load or Fetch Data")
print("=" * 80)
step_start = time.time()

CACHE_FILE = 'data/processed/research_lite_prices.parquet'
os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)

from src.utils.date_utils import calculate_start_date, calculate_start_date_str

END_DATE_DT = datetime.now()
END_DATE = END_DATE_DT.strftime('%Y%m%d')
START_DATE = calculate_start_date_str(END_DATE_DT, history_months=HISTORY_MONTHS_TARGET)

print("\n[Diagnostics] Date Range:")
print("  - Target history months: {}".format(HISTORY_MONTHS_TARGET))
print("  - END_DATE: {}".format(END_DATE))
print("  - START_DATE: {}".format(START_DATE))
print("  - Calculation method: timedelta(days=months*30)")

price_data = None
cache_used = False

if USE_CACHE and os.path.exists(CACHE_FILE):
    print("\n  - Loading from cache: {}".format(CACHE_FILE))
    price_data = pd.read_parquet(CACHE_FILE)
    print("  - [WARN] Using cached data - may not match target 100 stocks / 12 months")
    cache_used = True

if price_data is None or price_data['stock'].nunique() < 40:
    print("\n  - Fetching from AkShare...")

    BASE_STOCKS_RELIABLE = [
        '000001.SZ', '000002.SZ', '000858.SZ', '600000.SH', '600036.SH',
        '600028.SH', '600030.SH', '600050.SH', '600519.SH', '601318.SH',
        '601288.SH', '601398.SH', '601857.SH', '601988.SH', '000333.SZ',
        '000651.SZ', '002594.SZ', '600900.SH', '601012.SH', '601166.SH',
        '600036.SH', '600048.SH', '600104.SH', '600309.SH', '600436.SH',
        '600585.SH', '600690.SH', '600887.SH', '601088.SH', '601186.SH',
        '000725.SZ', '000898.SZ', '002415.SZ', '002475.SZ', '002594.SZ',
        '600009.SH', '600025.SH', '600100.SH', '600111.SH', '600150.SH',
        '600438.SH', '600760.SH', '600893.SH', '601066.SH', '601225.SH',
        '601236.SH', '601319.SH', '601336.SH', '601628.SH', '601668.SH',
        '000063.SZ', '000166.SZ', '000776.SZ', '002007.SZ', '002024.SZ',
        '002027.SZ', '002129.SZ', '002230.SZ', '002241.SZ', '002271.SZ',
        '002304.SZ', '002352.SZ', '002371.SZ', '002384.SZ', '002405.SZ',
        '002410.SZ', '002460.SZ', '002468.SZ', '002475.SZ', '002508.SZ',
        '002531.SZ', '002555.SZ', '002557.SZ', '002563.SZ', '002568.SZ',
        '600066.SH', '600089.SH', '600143.SH', '600153.SH', '600170.SH',
        '600177.SH', '600183.SH', '600196.SH', '600201.SH', '600219.SH',
        '600221.SH', '600233.SH', '600258.SH', '600271.SH', '600276.SH'
    ]

    import akshare as ak

    all_data = []
    fetched_symbols = []
    failed_symbols = []

    for i, stock_code in enumerate(BASE_STOCKS_RELIABLE[:STOCK_COUNT_TARGET + 20]):
        try:
            symbol = stock_code.split('.')[0]
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                start_date=START_DATE,
                end_date=END_DATE,
                adjust='qfq'
            )

            if df is not None and len(df) > 30:
                df = df.rename(columns={
                    '日期': 'date',
                    '开盘': 'open',
                    '收盘': 'close',
                    '最高': 'high',
                    '最低': 'low',
                    '成交量': 'volume',
                    '成交额': 'amount',
                    '涨跌幅': 'pct_change',
                    '换手率': 'turnover'
                })
                df['date'] = pd.to_datetime(df['date'])
                df['stock'] = stock_code
                all_data.append(df)
                fetched_symbols.append(stock_code)

            if (i + 1) % 20 == 0:
                print("  - Fetched {}/{} stocks... (success: {})".format(i + 1, STOCK_COUNT_TARGET + 20, len(fetched_symbols)))

            if len(fetched_symbols) >= STOCK_COUNT_TARGET:
                print("  - Target stock count ({}) reached!".format(STOCK_COUNT_TARGET))
                break

        except Exception as e:
            failed_symbols.append({
                'symbol': stock_code,
                'error': str(e)[:100]
            })

    print("\n[Diagnostics] Stock Fetch Results:")
    print("  - Target: {}".format(STOCK_COUNT_TARGET))
    print("  - Successfully fetched: {}".format(len(fetched_symbols)))
    print("  - Failed: {}".format(len(failed_symbols)))

    if len(failed_symbols) > 0:
        print("  - Failed symbols (first 5): {}".format([x['symbol'] for x in failed_symbols[:5]]))

    if len(fetched_symbols) == 0:
        print("  - [FAIL] Cannot fetch any real data")
        sys.exit(1)

    price_data = pd.concat(all_data, ignore_index=True)
    price_data = price_data.sort_values(['stock', 'date']).reset_index(drop=True)

    if USE_CACHE:
        price_data.to_parquet(CACHE_FILE, index=False)
        print("  - Saved to cache: {}".format(CACHE_FILE))

START_DATE = price_data['date'].min().strftime('%Y%m%d')
END_DATE = price_data['date'].max().strftime('%Y%m%d')
ACTUAL_STOCK_COUNT = price_data['stock'].nunique()

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
            'rank_ic_count': eval_result['rank_ic']['count']
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

X, metadata = dataset.get_samples()

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
print("  - metadata shape: {}".format(metadata.shape))
print("  - signal_date min/max: {} / {}".format(metadata['signal_date'].min(), metadata['signal_date'].max()))
print("  - target_start_date min/max: {} / {}".format(metadata['target_start_date'].min(), metadata['target_start_date'].max()))

alignment_violations = metadata[metadata['target_start_date'] <= metadata['signal_date']]
print("  - Violations (target_start_date <= signal_date): {}".format(len(alignment_violations)))
if len(alignment_violations) > 0:
    print("  - First 5 violations:")
    for idx, row in alignment_violations.head(5).iterrows():
        print("    - signal_date: {}, target_start_date: {}".format(row['signal_date'], row['target_start_date']))

checker = NeuralLeakageChecker()
leakage_results = checker.run_all_checks(
    columns=RAW_FEATURES,
    metadata=metadata,
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
report_lines.append("# Research Lite Pipeline Report (Reliability Audit)")
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

REPORT_FILE = 'reports/research_lite_report.md'
with open(REPORT_FILE, 'w', encoding='utf-8') as f:
    f.write(report)

print("\n  - Report saved: {}".format(REPORT_FILE))
print("  - [OK] Report generated in {:.2f}s".format(time.time() - step_start))

# ============================================================================
# Done
# ============================================================================
print("\n" + "=" * 80)
print("Research Lite Pipeline Completed!")
print("=" * 80)
print("Total time: {:.2f}s".format(time.time() - start_time))
print("Report: {}".format(REPORT_FILE))

if OVERALL_STATUS == 'FAIL':
    print("\n[IMPORTANT] Leakage check FAILED - results may be unreliable")
