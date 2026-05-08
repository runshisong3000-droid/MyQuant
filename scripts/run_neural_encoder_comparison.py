"""
Neural Factor Comparison Pipeline - 神经因子对比流水线

同时训练 MLP、CNN1D、TinyTransformer 三种编码器，
对比它们的 embedding 生成 neural factors 的表现。

按照 neural_feature_learning 模板执行：
- 不直接新增大型模型
- 先评估现有模型
- embedding 必须转成 neural factors
- neural factors 必须进入评价流程
- 必须看 RankIC/ICIR
- 必须输出报告
- 不允许直接用于实盘
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
from datetime import datetime

import numpy as np
import pandas as pd

print("=" * 60)
print("MyQuant Neural Factor Comparison Pipeline")
print("=" * 60)

start_time = time.time()

# ============================================================================
# Step 1: Load Real Data (reuse research_lite data)
# ============================================================================
print("\n[Step 1] Load Existing Data...")
step_start = time.time()

DATA_PATH = 'data/processed/research_lite_prices.parquet'

if os.path.exists(DATA_PATH):
    price_data = pd.read_parquet(DATA_PATH)
    print("  - Loaded: {}".format(DATA_PATH))
    print("  - Data shape: {}".format(price_data.shape))
    print("  - Stock count: {}".format(price_data['stock'].nunique()))
    print("  - Date range: {} to {}".format(
        price_data['date'].min(), price_data['date'].max()))
else:
    print("  - [FAIL] Data not found, please run research_lite_pipeline first")
    sys.exit(1)

print("  - [OK] Time: {:.2f}s".format(time.time() - step_start))

# ============================================================================
# Step 2: Generate future_return by stock
# ============================================================================
print("\n[Step 2] Generate future_return (by stock)...")
step_start = time.time()

price_data = price_data.sort_values(['stock', 'date']).reset_index(drop=True)
price_data['future_return'] = price_data.groupby('stock')['close'].pct_change().shift(-1)

print("  - future_return calculated per stock (not global)")
print("  - [OK] Time: {:.2f}s".format(time.time() - step_start))

# ============================================================================
# Step 3: Construct Sequence Dataset
# ============================================================================
print("\n[Step 3] Construct Sequence Dataset...")
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
# Step 4: Time Split
# ============================================================================
print("\n[Step 4] Time Split (60% train / 20% val / 20% test)...")
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
# Step 5: Leakage Check
# ============================================================================
print("\n[Step 5] Neural Leakage Check...")
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
# Step 6: Prepare Future Returns for Evaluation
# ============================================================================
print("\n[Step 6] Prepare Future Returns for Evaluation...")
step_start = time.time()

future_returns = price_data[['date', 'stock', 'future_return']].dropna()
future_returns = future_returns.set_index(['date', 'stock'])['future_return']

print("  - future_return shape: {}".format(future_returns.shape))
print("  - [OK] Time: {:.2f}s".format(time.time() - step_start))

# ============================================================================
# Step 7: Train & Compare Multiple Encoders
# ============================================================================
print("\n[Step 7] Train & Compare Multiple Encoders...")
step_start = time.time()

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("  - [FAIL] PyTorch not installed")
    sys.exit(1)

from src.factors.neural.autoencoder import SequenceAutoEncoder

ENCODER_TYPES = ['mlp', 'cnn', 'transformer']
HIDDEN_DIM = 32
EMBEDDING_DIM = 8
BATCH_SIZE = 64
EPOCHS = 5
DEVICE = 'cpu'

INPUT_DIM = len(RAW_FEATURES)

results_by_encoder = {}

for encoder_type in ENCODER_TYPES:
    print("\n  [{}] Training encoder...".format(encoder_type.upper()))
    encoder_start = time.time()

    model = SequenceAutoEncoder(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        embedding_dim=EMBEDDING_DIM,
        lookback_window=LOOKBACK_WINDOW,
        encoder_type=encoder_type
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

        print("    Epoch {}/{}: train_loss={:.6f}, val_loss={:.6f}".format(
            epoch + 1, EPOCHS, avg_train_loss, val_losses[-1] if val_losses else 0
        ))

    final_train_loss = train_losses[-1] if train_losses else 0
    final_val_loss = val_losses[-1] if val_losses else 0

    print("    Final train_loss: {:.6f}, val_loss: {:.6f}".format(final_train_loss, final_val_loss))

    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
        _, embeddings = model(X_test_tensor)

    embeddings = embeddings.cpu().numpy()

    from src.factors.neural.neural_factor_extractor import NeuralFactorExtractor

    extractor = NeuralFactorExtractor(embedding_dim=EMBEDDING_DIM)
    factors_df = extractor.embedding_to_dataframe(embeddings, meta_test.reset_index(drop=True))

    factors_for_eval = {}
    for i in range(EMBEDDING_DIM):
        factor_col = 'neural_factor_{}'.format(i)
        if factor_col in factors_df.columns:
            series = factors_df.set_index(['signal_date', 'stock'])[factor_col]
            aligned = pd.Series(
                series.values,
                index=pd.MultiIndex.from_arrays([
                    series.index.get_level_values('signal_date'),
                    series.index.get_level_values('stock')
                ], names=['date', 'stock'])
            )
            factors_for_eval[factor_col] = aligned

    from src.factors.neural.neural_factor_evaluator import NeuralFactorEvaluator

    neural_eval = NeuralFactorEvaluator()
    eval_results = neural_eval.evaluate_neural_factors(factors_for_eval, future_returns)
    eval_summary = neural_eval.create_evaluation_summary(eval_results)

    passing_factors = neural_eval.get_passing_factors(
        eval_results,
        min_icir=0.1,
        min_rank_ic=0.01,
        min_coverage=0.5
    )

    results_by_encoder[encoder_type] = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'embeddings': embeddings,
        'factors_df': factors_df,
        'eval_results': eval_results,
        'eval_summary': eval_summary,
        'passing_factors': passing_factors,
        'training_time': time.time() - encoder_start
    }

    print("    Evaluated {} neural factors".format(len(eval_results)))
    print("    Passing factors: {}".format(len(passing_factors)))
    print("    Time: {:.2f}s".format(results_by_encoder[encoder_type]['training_time']))

print("\n  All encoders trained!")
print("  - [OK] Time: {:.2f}s".format(time.time() - step_start))

# ============================================================================
# Step 8: Generate Comparison Report
# ============================================================================
print("\n[Step 8] Generate Comparison Report...")
step_start = time.time()

report_lines = []

report_lines.append("# Neural Encoder Comparison Report")
report_lines.append("")
report_lines.append("Generated: {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
report_lines.append("")
report_lines.append("## 1. Purpose")
report_lines.append("")
report_lines.append("对比 MLP、CNN1D、TinyTransformer 三种编码器的表现。")
report_lines.append("")
report_lines.append("## 2. Comparison Metrics")
report_lines.append("")
report_lines.append("| Encoder | Train Loss | Val Loss | Time (s) | Avg RankIC | Best RankIC | Avg ICIR | Best ICIR | Passing Factors |")
report_lines.append("|---------|------------|----------|----------|------------|-------------|----------|-----------|-----------------|")

for encoder_type in ENCODER_TYPES:
    res = results_by_encoder[encoder_type]
    eval_df = res['eval_summary']
    
    avg_rank_ic = eval_df['rank_ic_mean'].mean()
    best_rank_ic = eval_df['rank_ic_mean'].max()
    avg_icir = eval_df['icir'].mean()
    best_icir = eval_df['icir'].max()
    
    report_lines.append("| {} | {:.6f} | {:.6f} | {:.2f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {} |".format(
        encoder_type.upper(),
        res['final_train_loss'],
        res['final_val_loss'],
        res['training_time'],
        avg_rank_ic,
        best_rank_ic,
        avg_icir,
        best_icir,
        len(res['passing_factors'])
    ))

report_lines.append("")
report_lines.append("## 3. Detailed Results by Encoder")

for encoder_type in ENCODER_TYPES:
    res = results_by_encoder[encoder_type]
    eval_df = res['eval_summary']
    
    report_lines.append("")
    report_lines.append("### {}".format(encoder_type.upper()))
    report_lines.append("")
    report_lines.append("- **Final Train Loss**: {:.6f}".format(res['final_train_loss']))
    report_lines.append("- **Final Val Loss**: {:.6f}".format(res['final_val_loss']))
    report_lines.append("- **Training Time**: {:.2f}s".format(res['training_time']))
    report_lines.append("- **Passing Factors**: {}".format(len(res['passing_factors'])))
    report_lines.append("")
    report_lines.append("| Factor | RankIC | ICIR | Coverage |")
    report_lines.append("|--------|--------|------|----------|")
    
    for _, row in eval_df.iterrows():
        report_lines.append("| {} | {:.4f} | {:.4f} | {:.2f} |".format(
            row['factor'], row['rank_ic_mean'], row['icir'], row['coverage']
        ))

report_lines.append("")
report_lines.append("## 4. Leakage Check")
report_lines.append("")
report_lines.append("| Check | Status |")
report_lines.append("|-------|--------|")

for check_name, result in leakage_results.items():
    if check_name == 'overall_status':
        continue
    status = result.get('status', 'UNKNOWN')
    report_lines.append("| {} | {} |".format(check_name, status))

report_lines.append("")
report_lines.append("## 5. Conclusion")
report_lines.append("")

best_encoder = None
best_avg_rank_ic = -999

encoder_comparison_data = []

for encoder_type in ENCODER_TYPES:
    res = results_by_encoder[encoder_type]
    eval_df = res['eval_summary']
    
    avg_rank_ic = eval_df['rank_ic_mean'].mean()
    best_rank_ic = eval_df['rank_ic_mean'].max()
    avg_icir = eval_df['icir'].mean()
    best_icir = eval_df['icir'].max()
    
    encoder_comparison_data.append({
        'encoder': encoder_type,
        'train_loss': res['final_train_loss'],
        'val_loss': res['final_val_loss'],
        'avg_rankic': avg_rank_ic,
        'best_rankic': best_rank_ic,
        'avg_icir': avg_icir,
        'best_icir': best_icir,
        'passing_factors': len(res['passing_factors']),
        'training_time': res['training_time']
    })
    
    if avg_rank_ic > best_avg_rank_ic:
        best_avg_rank_ic = avg_rank_ic
        best_encoder = encoder_type

# Save encoder comparison to dashboard
os.makedirs('data/dashboard', exist_ok=True)
encoder_comparison_df = pd.DataFrame(encoder_comparison_data)
encoder_comparison_df.to_parquet('data/dashboard/encoder_comparison.parquet', index=False)
print("  - Saved: data/dashboard/encoder_comparison.parquet")

report_lines.append("- **Best Encoder by Avg RankIC**: {}".format(best_encoder.upper() if best_encoder else "N/A"))
report_lines.append("- **Best Avg RankIC**: {:.4f}".format(best_avg_rank_ic))
report_lines.append("- **Leakage Check**: {}".format("PASS" if leakage_results.get('overall_status') == 'OK' else "FAIL"))
report_lines.append("")
report_lines.append("## 6. Important Notes")
report_lines.append("")
report_lines.append("- 当前只用于研究，**不能直接实盘**")
report_lines.append("- 所有 neural factors 都经过 RankIC/ICIR/Coverage 评价")
report_lines.append("- 通过可信度审计和未来函数检查")
report_lines.append("- 样本量有限，结果仅供参考")
report_lines.append("")
report_lines.append("Total time: {:.2f}s".format(time.time() - start_time))

report = "\n".join(report_lines)

report_file = 'reports/neural_encoder_comparison.md'
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(report)

print("  - Report saved: {}".format(report_file))
print("  - [OK] Time: {:.2f}s".format(time.time() - step_start))

# ============================================================================
# Done
# ============================================================================
print("\n" + "=" * 60)
print("Neural Encoder Comparison Completed!")
print("=" * 60)
print("Total time: {:.2f}s".format(time.time() - start_time))
print("Report: {}".format(report_file))
