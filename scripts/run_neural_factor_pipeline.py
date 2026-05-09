"""
Neural Factor Pipeline - 神经因子流水线

支持多种 profile：
- student_laptop: 20只股票，6个月
- research_lite: 100只股票，12个月
- research_medium_trial: 150只股票，18个月
- research_medium: 300只股票，24个月

流程:
1. 加载真实数据（从 profile-specific 目录）
2. 数据标准化
3. 生成future_return(按stock分组)
4. 构造序列数据集
5. 时间切分
6. 未来函数检测
7. 训练AutoEncoder
8. 提取embedding
9. 转换为neural factors
10. 评价neural factors
11. 对比普通因子
12. 输出报告（到 profile-specific 目录）
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
parser = argparse.ArgumentParser(description='Neural Factor Pipeline')
parser.add_argument('--profile', default='research_lite', 
                    help='Profile name from compute_profile.yaml')
parser.add_argument('--force', action='store_true',
                    help='Force re-run even if artifacts exist')
args = parser.parse_args()

profile_name = args.profile
force_run = args.force

print("=" * 60)
print("MyQuant Neural Factor Pipeline")
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
# Step 1: Load Real Data (从 profile-specific 目录读取)
# ============================================================================
print("\n[Step 1] Load Real Data...")
step_start = time.time()

PRICES_PATH = os.path.join(PROFILE_DIR, 'prices.parquet')
price_data = None
data_source = profile_name

if os.path.exists(PRICES_PATH):
    print("  - Loading from: {}".format(PRICES_PATH))
    price_data = pd.read_parquet(PRICES_PATH)
    print("  - Loaded: shape={}".format(price_data.shape))
    print("  - Date range: {} to {}".format(
        price_data['date'].min().date() if 'date' in price_data.columns else 'N/A',
        price_data['date'].max().date() if 'date' in price_data.columns else 'N/A'
    ))
    print("  - Stock count: {}".format(price_data['stock'].nunique() if 'stock' in price_data.columns else 'N/A'))
else:
    print("  - [FAIL] Prices not found at: {}".format(PRICES_PATH))
    print("  - Please run run_research_lite_pipeline.py first")
    sys.exit(1)

print("  - [OK] Time: {:.2f}s".format(time.time() - step_start))

# ============================================================================
# Step 2: Data Standardization
# ============================================================================
print("\n[Step 2] Data Standardization...")
step_start = time.time()

standard_columns = ['date', 'stock', 'open', 'high', 'low', 'close', 'volume', 'amount', 'turnover']

for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'turnover']:
    if col not in price_data.columns:
        price_data[col] = 0

price_data = price_data[standard_columns]

price_data = price_data.dropna(subset=['close', 'volume'])

print("  - Standard columns: {}".format(standard_columns))
print("  - Data shape after cleaning: {}".format(price_data.shape))
print("  - [OK] Time: {:.2f}s".format(time.time() - step_start))

# ============================================================================
# Step 3: Generate future_return by stock
# ============================================================================
print("\n[Step 3] Generate future_return (by stock)...")
step_start = time.time()

price_data = price_data.sort_values(['stock', 'date']).reset_index(drop=True)

price_data['daily_return'] = price_data.groupby('stock')['close'].pct_change()

price_data['future_return_1d'] = price_data.groupby('stock')['close'].pct_change().shift(-1)

print("  - future_return calculated per stock (not global)")
print("  - daily_return: t-1 to t return")
print("  - future_return_1d: t to t+1 return")
print("  - [OK] Time: {:.2f}s".format(time.time() - step_start))

# ============================================================================
# Step 4: Construct Sequence Dataset
# ============================================================================
print("\n[Step 4] Construct Sequence Dataset...")
step_start = time.time()

from src.factors.neural.sequence_dataset import SequenceDataset

LOOKBACK_WINDOW = 20
RAW_FEATURES = ['open', 'high', 'low', 'close', 'volume']
TARGET_HORIZON = 1

dataset = SequenceDataset(
    df=price_data,
    lookback_window=LOOKBACK_WINDOW,
    features=RAW_FEATURES,
    target_horizon=TARGET_HORIZON
)

X, metadata = dataset.get_samples()

print("  - lookback_window: {}".format(LOOKBACK_WINDOW))
print("  - raw_features: {}".format(RAW_FEATURES))
print("  - target_horizon: {}".format(TARGET_HORIZON))
print("  - X shape: {}".format(X.shape))
print("  - metadata shape: {}".format(metadata.shape))
print("  - [OK] Time: {:.2f}s".format(time.time() - step_start))

# ============================================================================
# Step 5: Time Split
# ============================================================================
print("\n[Step 5] Time Split (60% train / 20% val / 20% test)...")
step_start = time.time()

TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

splits = dataset.get_train_val_test_split(
    train_ratio=TRAIN_RATIO,
    val_ratio=VAL_RATIO,
    test_ratio=TEST_RATIO
)

X_train, meta_train = splits['train']
X_val, meta_val = splits['val']
X_test, meta_test = splits['test']

print("  - Train samples: {} ({} to {})".format(
    len(X_train),
    meta_train['signal_date'].min().date() if len(meta_train) > 0 else 'N/A',
    meta_train['signal_date'].max().date() if len(meta_train) > 0 else 'N/A'
))
print("  - Validation samples: {} ({} to {})".format(
    len(X_val),
    meta_val['signal_date'].min().date() if len(meta_val) > 0 else 'N/A',
    meta_val['signal_date'].max().date() if len(meta_val) > 0 else 'N/A'
))
print("  - Test samples: {} ({} to {})".format(
    len(X_test),
    meta_test['signal_date'].min().date() if len(meta_test) > 0 else 'N/A',
    meta_test['signal_date'].max().date() if len(meta_test) > 0 else 'N/A'
))

if len(X_train) < 50:
    print("  - [WARN] Insufficient training samples!")
    print("  - [WARN] Pipeline will continue but results may not be reliable.")

print("  - [OK] Time: {:.2f}s".format(time.time() - step_start))

# ============================================================================
# Step 6: Leakage Check
# ============================================================================
print("\n[Step 6] Neural Leakage Check...")
step_start = time.time()

from src.validation.neural_leakage_check import NeuralLeakageChecker

checker = NeuralLeakageChecker()

leakage_results = checker.run_all_checks(
    columns=RAW_FEATURES,
    metadata=metadata,
    train_dates=(meta_train['signal_date'].min(), meta_train['signal_date'].max()) if len(meta_train) > 0 else (None, None),
    val_dates=(meta_val['signal_date'].min(), meta_val['signal_date'].max()) if len(meta_val) > 0 else (None, None),
    test_dates=(meta_test['signal_date'].min(), meta_test['signal_date'].max()) if len(meta_test) > 0 else (None, None)
)

print("  - Overall leakage status: {}".format(leakage_results.get('overall_status', 'UNKNOWN')))

for check_name, result in leakage_results.items():
    if check_name == 'overall_status':
        continue
    status = result.get('status', 'UNKNOWN')
    print("    {}: {}".format(check_name, status))

if leakage_results.get('overall_status') == 'FAIL':
    print("  - [FAIL] Leakage check failed, pipeline will stop.")
    sys.exit(1)

print("  - [OK] Time: {:.2f}s".format(time.time() - step_start))

# ============================================================================
# Step 7: Train AutoEncoder
# ============================================================================
print("\n[Step 7] Train SequenceAutoEncoder...")
step_start = time.time()

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("  - [FAIL] PyTorch not installed")
    sys.exit(1)

from src.factors.neural.autoencoder import SequenceAutoEncoder

HIDDEN_DIM = 32
EMBEDDING_DIM = 8
BATCH_SIZE = 64
EPOCHS = 5
DEVICE = 'cpu'

INPUT_DIM = len(RAW_FEATURES)

model = SequenceAutoEncoder(
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    embedding_dim=EMBEDDING_DIM,
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
val_losses = []

if len(X_val) > 0:
    X_val_tensor = torch.FloatTensor(X_val)

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

    avg_train_loss = epoch_loss / n_batches if n_batches > 0 else 0
    train_losses.append(avg_train_loss)

    if len(X_val) > 0:
        model.eval()
        with torch.no_grad():
            val_recon, _ = model(X_val_tensor.to(DEVICE))
            val_loss = criterion(val_recon, X_val_tensor.to(DEVICE)).item()
            val_losses.append(val_loss)
        model.train()

    print("  - Epoch {}/{}: train_loss={:.6f}, val_loss={:.6f}".format(
        epoch + 1, EPOCHS,
        avg_train_loss,
        val_losses[-1] if val_losses else 0
    ))

print("  - Final train_loss: {:.6f}".format(train_losses[-1] if train_losses else 0))
print("  - [OK] Time: {:.2f}s".format(time.time() - step_start))

# ============================================================================
# Step 8: Extract Embeddings (All samples, not just test!)
# ============================================================================
print("\n[Step 8] Extract Embeddings (All samples)...")
step_start = time.time()

model.eval()

with torch.no_grad():
    X_all_tensor = torch.FloatTensor(X).to(DEVICE)
    _, embeddings = model(X_all_tensor)

embeddings = embeddings.cpu().numpy()

print("  - Embedding shape: {}".format(embeddings.shape))
print("  - [OK] Time: {:.2f}s".format(time.time() - step_start))

# ============================================================================
# Step 9: Convert to Neural Factors
# ============================================================================
print("\n[Step 9] Convert to Neural Factors...")
step_start = time.time()

from src.factors.neural.neural_factor_extractor import NeuralFactorExtractor

extractor = NeuralFactorExtractor(embedding_dim=EMBEDDING_DIM)

factors_df = extractor.embedding_to_dataframe(embeddings, metadata.reset_index(drop=True))

print("  - Neural factors shape: {}".format(factors_df.shape))
print("  - Columns: {}".format(list(factors_df.columns[:5])))

os.makedirs('data/factors', exist_ok=True)
os.makedirs('reports', exist_ok=True)

extractor.save_factors(factors_df, 'data/factors/neural_factors.parquet')

# Also save to profile-specific dashboard directory
factors_df.to_parquet(os.path.join(DASHBOARD_DIR, 'neural_factors.parquet'), index=False)
print(f"  - Saved to dashboard: {os.path.join(DASHBOARD_DIR, 'neural_factors.parquet')}")

# Save neural factor summary (will be generated later)
# We save this part after evaluation

# Save neural factor metadata to profile-specific dashboard directory
metadata_dict = {
    'data_source': data_source,
    'price_panel_path': PRICES_PATH,
    'input_date_range': (price_data['date'].min(), price_data['date'].max()),
    'output_date_range': (factors_df['signal_date'].min(), factors_df['signal_date'].max()) if 'signal_date' in factors_df.columns else (None, None),
    'input_stock_count': price_data['stock'].nunique(),
    'output_stock_count': factors_df['stock'].nunique() if 'stock' in factors_df.columns else 0,
    'sample_count': len(factors_df),
    'train_start': meta_train['signal_date'].min() if len(meta_train) > 0 else None,
    'train_end': meta_train['signal_date'].max() if len(meta_train) > 0 else None,
    'validation_start': meta_val['signal_date'].min() if len(meta_val) > 0 else None,
    'validation_end': meta_val['signal_date'].max() if len(meta_val) > 0 else None,
    'test_start': meta_test['signal_date'].min() if len(meta_test) > 0 else None,
    'test_end': meta_test['signal_date'].max() if len(meta_test) > 0 else None,
    'leakage_check_status': leakage_results.get('overall_status', 'UNKNOWN'),
    'model_type': 'SequenceAutoEncoder',
    'encoder_type': 'MLP',
    'lookback_window': LOOKBACK_WINDOW,
    'embedding_dim': EMBEDDING_DIM,
    'hidden_dim': HIDDEN_DIM,
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE
}
import json
neural_factor_metadata_path = os.path.join(DASHBOARD_DIR, 'neural_factor_metadata.json')
with open(neural_factor_metadata_path, 'w', encoding='utf-8') as f:
    json.dump({k: str(v) for k, v in metadata_dict.items()}, f, ensure_ascii=False, indent=2)
print(f"  - Saved to dashboard: {neural_factor_metadata_path}")

extractor.save_metadata(metadata_dict, 'reports/neural_factor_metadata.json')

print("  - [OK] Time: {:.2f}s".format(time.time() - step_start))

# ============================================================================
# Step 10: Evaluate Neural Factors (with diagnostics)
# ============================================================================
print("\n[Step 10] Evaluate Neural Factors (with Diagnostics)...")
step_start = time.time()

print("\n  [DIAGNOSTIC] Index Alignment Check:")
print("  - factors_df shape: {}".format(factors_df.shape))
print("  - factors_df index type: {}".format(type(factors_df.index)))

if isinstance(factors_df.index, pd.MultiIndex):
    print("  - factors_df is MultiIndex")
    print("  - factors_df index levels: {}".format(factors_df.index.names))
else:
    print("  - factors_df is NOT MultiIndex, resetting...")
    factors_df = factors_df.reset_index(drop=True)

print("  - meta_test signal_date range: {} to {}".format(
    meta_test['signal_date'].min(), meta_test['signal_date'].max()))
print("  - meta_test stock unique: {}".format(meta_test['stock'].nunique()))

print("\n  [DIAGNOSTIC] Creating future_return with proper alignment:")

df_with_future = price_data.copy()
df_with_future['future_return'] = df_with_future.groupby('stock')['close'].pct_change().shift(-TARGET_HORIZON)

future_returns = df_with_future[['date', 'stock', 'future_return']].dropna()
future_returns = future_returns.set_index(['date', 'stock'])['future_return']

print("  - future_return shape: {}".format(future_returns.shape))
print("  - future_return index type: {}".format(type(future_returns.index)))

if not isinstance(future_returns.index, pd.MultiIndex):
    print("  - [WARN] future_return is not MultiIndex, creating...")

print("  - future_return shape after fix: {}".format(future_returns.shape))

print("\n  [DIAGNOSTIC] Creating properly aligned neural factor Series:")

factors_for_eval = {}
for i in range(EMBEDDING_DIM):
    factor_col = 'neural_factor_{}'.format(i)
    if factor_col in factors_df.columns:
        factors_for_eval[factor_col] = factors_df.set_index(['signal_date', 'stock'])[factor_col]

print("  - Created {} neural factor Series".format(len(factors_for_eval)))

if factors_for_eval:
    first_factor = list(factors_for_eval.values())[0]
    print("  - First factor index type: {}".format(type(first_factor.index)))
    print("  - First factor index names: {}".format(first_factor.index.names if hasattr(first_factor.index, 'names') else 'N/A'))
    print("  - First factor shape: {}".format(first_factor.shape))

    print("\n  [DIAGNOSTIC] Index intersection check:")
    common_idx = first_factor.index.intersection(future_returns.index)
    print("  - Common index count: {}".format(len(common_idx)))

    if len(common_idx) == 0:
        print("  - [WARN] No common index! Trying to align...")

        aligned_factors = {}
        for name, series in factors_for_eval.items():
            aligned = pd.Series(
                series.values,
                index=pd.MultiIndex.from_arrays([
                    series.index.get_level_values('signal_date'),
                    series.index.get_level_values('stock')
                ], names=['date', 'stock'])
            )
            aligned_factors[name] = aligned

        factors_for_eval = aligned_factors
        common_idx = list(factors_for_eval.values())[0].index.intersection(future_returns.index)
        print("  - After alignment, common index count: {}".format(len(common_idx)))

print("  - [OK] Diagnostic complete, starting evaluation")

from src.factors.neural.neural_factor_evaluator import NeuralFactorEvaluator

neural_eval = NeuralFactorEvaluator()
eval_results = neural_eval.evaluate_neural_factors(factors_for_eval, future_returns)

print("  - Evaluated {} neural factors".format(len(eval_results)))

eval_summary = neural_eval.create_evaluation_summary(eval_results)
print("\n  Neural Factor Evaluation:")
for _, row in eval_summary.iterrows():
    print("    {}: IC={:.4f}, ICIR={:.4f}, coverage={:.2f}".format(
        row['factor'], row['rank_ic_mean'], row['icir'], row['coverage']
    ))

print("  - [OK] Time: {:.2f}s".format(time.time() - step_start))

# ============================================================================
# Step 11: Screen Neural Factors
# ============================================================================
print("\n[Step 11] Screen Neural Factors...")
step_start = time.time()

passing_factors = neural_eval.get_passing_factors(
    eval_results,
    min_icir=0.1,
    min_rank_ic=0.01,
    min_coverage=0.5
)

print("  - Passing factors: {}".format(len(passing_factors)))
for f in passing_factors:
    print("    - {}".format(f))

print("  - [OK] Time: {:.2f}s".format(time.time() - step_start))

# ============================================================================
# Step 12: Generate Report
# ============================================================================
print("\n[Step 12] Generate Report...")
step_start = time.time()

report_lines = []

report_lines.append("# Neural Feature Learning Report")
report_lines.append("")
report_lines.append("Generated: {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
report_lines.append("")
report_lines.append("## 1. Purpose")
report_lines.append("")
report_lines.append("从原始 OHLCV 序列中学习 latent representation，并转化为 neural factors。")
report_lines.append("")
report_lines.append("## 2. Current MyQuant Integration")
report_lines.append("")
report_lines.append("复用模块:")
report_lines.append("- run_student_laptop_pipeline.py 数据加载逻辑")
report_lines.append("- FactorEvaluator")
report_lines.append("- FactorScreener")
report_lines.append("- FactorGatekeeper")
report_lines.append("- Reliability Audit 规则")
report_lines.append("")
report_lines.append("## 3. Data Overview")
report_lines.append("")
report_lines.append("- **Data Source**: AkShare")
report_lines.append("- **Stock Count**: {}".format(price_data['stock'].nunique()))
report_lines.append("- **Time Range**: {} ~ {}".format(price_data['date'].min(), price_data['date'].max()))
report_lines.append("- **Raw Data Shape**: {}".format(price_data.shape))
report_lines.append("- **Sequence Sample Shape**: X={}, metadata={}".format(X.shape, metadata.shape))
report_lines.append("- **Raw Features**: {}".format(RAW_FEATURES))
report_lines.append("")
report_lines.append("## 4. Model Overview")
report_lines.append("")
report_lines.append("- **Model Type**: SequenceAutoEncoder")
report_lines.append("- **Encoder Type**: MLP")
report_lines.append("- **lookback_window**: {}".format(LOOKBACK_WINDOW))
report_lines.append("- **embedding_dim**: {}".format(EMBEDDING_DIM))
report_lines.append("- **hidden_dim**: {}".format(HIDDEN_DIM))
report_lines.append("- **epochs**: {}".format(EPOCHS))
report_lines.append("- **batch_size**: {}".format(BATCH_SIZE))
report_lines.append("- **device**: {}".format(DEVICE))
report_lines.append("- **loss_function**: MSELoss")
report_lines.append("")
report_lines.append("## 5. Time Split")
report_lines.append("")
report_lines.append("- **Train Period**: {} ~ {} ({} samples)".format(
    meta_train['signal_date'].min().date() if len(meta_train) > 0 else 'N/A',
    meta_train['signal_date'].max().date() if len(meta_train) > 0 else 'N/A',
    len(X_train)
))
report_lines.append("- **Validation Period**: {} ~ {} ({} samples)".format(
    meta_val['signal_date'].min().date() if len(meta_val) > 0 else 'N/A',
    meta_val['signal_date'].max().date() if len(meta_val) > 0 else 'N/A',
    len(X_val)
))
report_lines.append("- **Test Period**: {} ~ {} ({} samples)".format(
    meta_test['signal_date'].min().date() if len(meta_test) > 0 else 'N/A',
    meta_test['signal_date'].max().date() if len(meta_test) > 0 else 'N/A',
    len(X_test)
))
report_lines.append("")
report_lines.append("## 6. Leakage Check")
report_lines.append("")
report_lines.append("| Check | Status |")
report_lines.append("|-------|--------|")

for check_name, result in leakage_results.items():
    if check_name == 'overall_status':
        continue
    status = result.get('status', 'UNKNOWN')
    report_lines.append("| {} | {} |".format(check_name, status))

report_lines.append("")
report_lines.append("## 7. Training Result")
report_lines.append("")
report_lines.append("- **Final Train Loss**: {:.6f}".format(train_losses[-1] if train_losses else 0))
report_lines.append("- **Final Val Loss**: {:.6f}".format(val_losses[-1] if val_losses else 0))
report_lines.append("- **Loss Curve**: Not saved (small model)")
report_lines.append("")
report_lines.append("## 8. Neural Factor Evaluation")
report_lines.append("")
report_lines.append("| Factor | RankIC | ICIR | Coverage | Status |")
report_lines.append("|--------|--------|------|----------|--------|")

for _, row in eval_summary.iterrows():
    report_lines.append("| {} | {:.4f} | {:.4f} | {:.2f} | {} |".format(
        row['factor'], row['rank_ic_mean'], row['icir'], row['coverage'], row['status']
    ))

report_lines.append("")
report_lines.append("## 9. Comparison With Existing Formula Factors")
report_lines.append("")
report_lines.append("Formula factors (from student_laptop_report.md):")
report_lines.append("- momentum_simple_20: IC=0.0510, ICIR=0.1536")
report_lines.append("- momentum_rel_ma20: IC=0.0373, ICIR=0.1168")
report_lines.append("- momentum_simple_10: IC=0.0337, ICIR=0.1013")
report_lines.append("")
report_lines.append("Neural factors (this run):")

for _, row in eval_summary.head(5).iterrows():
    report_lines.append("- {}: IC={:.4f}, ICIR={:.4f}".format(
        row['factor'], row['rank_ic_mean'], row['icir']
    ))

report_lines.append("")
report_lines.append("## 10. Limitations")
report_lines.append("")
report_lines.append("- 样本只有 {} 只股票".format(price_data['stock'].nunique()))
report_lines.append("- 时间只有约 6 个月")
report_lines.append("- 停牌/涨跌停过滤仍未完善")
report_lines.append("- Neural factor 不能直接实盘")
report_lines.append("- 当前只用于工程验证和初步研究")
report_lines.append("")
report_lines.append("## 11. Conclusion")
report_lines.append("")
report_lines.append("1. Neural feature pipeline: {}".format("RUN SUCCESS" if len(eval_results) > 0 else "RUN FAILED"))
report_lines.append("2. Future leakage detected: {}".format("YES - must review" if leakage_results.get('overall_status') != 'OK' else "NO"))
report_lines.append("3. Neural factors have research value: {}".format("MAYBE - need more validation" if len(passing_factors) > 0 else "NOT YET"))
report_lines.append("4. Can proceed to longer sample validation: {}".format("NOT YET - need larger sample"))
report_lines.append("5. NOT ready for live trading")
report_lines.append("")
report_lines.append("Total time: {:.2f}s".format(time.time() - start_time))

report = "\n".join(report_lines)

report_file = 'reports/neural_factor_report.md'
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(report)

print("  - Report saved: {}".format(report_file))
print("  - [OK] Time: {:.2f}s".format(time.time() - step_start))

# ============================================================================
# Done
# ============================================================================
print("\n" + "=" * 60)
print("Neural Factor Pipeline Completed!")
print("=" * 60)
print("Total time: {:.2f}s".format(time.time() - start_time))
print("Report: {}".format(report_file))
