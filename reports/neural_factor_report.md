# Neural Feature Learning Report

Generated: 2026-05-07 23:25:30

## 1. Purpose

从原始 OHLCV 序列中学习 latent representation，并转化为 neural factors。

## 2. Current MyQuant Integration

复用模块:
- run_student_laptop_pipeline.py 数据加载逻辑
- FactorEvaluator
- FactorScreener
- FactorGatekeeper
- Reliability Audit 规则

## 3. Data Overview

- **Data Source**: AkShare
- **Stock Count**: 20
- **Time Range**: 20260107 ~ 20260507
- **Raw Data Shape**: (1540, 11)
- **Sequence Sample Shape**: X=(1120, 20, 5), metadata=(1120, 5)
- **Raw Features**: ['open', 'high', 'low', 'close', 'volume']

## 4. Model Overview

- **Model Type**: SequenceAutoEncoder
- **Encoder Type**: MLP
- **lookback_window**: 20
- **embedding_dim**: 8
- **hidden_dim**: 32
- **epochs**: 5
- **batch_size**: 64
- **device**: cpu
- **loss_function**: MSELoss

## 5. Time Split

- **Train Period**: 2026-02-03 ~ 2026-03-27 (660 samples)
- **Validation Period**: 2026-03-30 ~ 2026-04-14 (220 samples)
- **Test Period**: 2026-04-15 ~ 2026-04-30 (240 samples)

## 6. Leakage Check

| Check | Status |
|-------|--------|
| feature_columns | OK |
| sequence_dates | OK |
| target_alignment | OK |
| scaler_fit_scope | OK |

## 7. Training Result

- **Final Train Loss**: 263143339101.090912
- **Final Val Loss**: 132992589824.000000
- **Loss Curve**: Not saved (small model)

## 8. Neural Factor Evaluation

| Factor | RankIC | ICIR | Coverage | Status |
|--------|--------|------|----------|--------|
| neural_factor_0 | 0.0507 | 0.2443 | 1.00 | evaluated |
| neural_factor_7 | 0.0490 | 0.2368 | 1.00 | evaluated |
| neural_factor_1 | 0.0479 | 0.2352 | 1.00 | evaluated |
| neural_factor_2 | 0.0422 | 0.2054 | 1.00 | evaluated |
| neural_factor_5 | -0.0435 | 0.2083 | 1.00 | evaluated |
| neural_factor_6 | -0.0497 | 0.2401 | 1.00 | evaluated |
| neural_factor_4 | -0.0499 | 0.2479 | 1.00 | evaluated |
| neural_factor_3 | -0.0506 | 0.2497 | 1.00 | evaluated |

## 9. Comparison With Existing Formula Factors

Formula factors (from student_laptop_report.md):
- momentum_simple_20: IC=0.0510, ICIR=0.1536
- momentum_rel_ma20: IC=0.0373, ICIR=0.1168
- momentum_simple_10: IC=0.0337, ICIR=0.1013

Neural factors (this run):
- neural_factor_0: IC=0.0507, ICIR=0.2443
- neural_factor_7: IC=0.0490, ICIR=0.2368
- neural_factor_1: IC=0.0479, ICIR=0.2352
- neural_factor_2: IC=0.0422, ICIR=0.2054
- neural_factor_5: IC=-0.0435, ICIR=0.2083

## 10. Limitations

- 样本只有 20 只股票
- 时间只有约 6 个月
- 停牌/涨跌停过滤仍未完善
- Neural factor 不能直接实盘
- 当前只用于工程验证和初步研究

## 11. Conclusion

1. Neural feature pipeline: RUN SUCCESS
2. Future leakage detected: NO
3. Neural factors have research value: MAYBE - need more validation
4. Can proceed to longer sample validation: NOT YET - need larger sample
5. NOT ready for live trading

Total time: 65.95s