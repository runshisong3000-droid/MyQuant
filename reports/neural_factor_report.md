# Neural Feature Learning Report

Generated: 2026-05-09 19:32:29

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
- **Stock Count**: 300
- **Time Range**: 2024-05-20 00:00:00 ~ 2026-05-08 00:00:00
- **Raw Data Shape**: (142830, 11)
- **Sequence Sample Shape**: X=(136530, 20, 5), metadata=(136530, 5)
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

- **Train Period**: 2024-06-17 ~ 2025-07-29 (81738 samples)
- **Validation Period**: 2025-07-30 ~ 2025-12-11 (27266 samples)
- **Test Period**: 2025-12-12 ~ 2026-05-06 (27526 samples)

## 6. Leakage Check

| Check | Status |
|-------|--------|
| feature_columns | OK |
| sequence_dates | OK |
| target_alignment | OK |
| scaler_fit_scope | OK |

## 7. Training Result

- **Final Train Loss**: 0.025839
- **Final Val Loss**: 0.029402
- **Loss Curve**: Not saved (small model)

## 8. Neural Factor Evaluation

| Factor | RankIC | ICIR | Coverage | Status |
|--------|--------|------|----------|--------|
| neural_factor_6 | 0.0455 | 0.2654 | 1.00 | evaluated |
| neural_factor_3 | 0.0221 | 0.2082 | 1.00 | evaluated |
| neural_factor_2 | 0.0132 | 0.0851 | 1.00 | evaluated |
| neural_factor_7 | 0.0069 | 0.0454 | 1.00 | evaluated |
| neural_factor_1 | -0.0013 | 0.0095 | 1.00 | evaluated |
| neural_factor_4 | -0.0045 | 0.0292 | 1.00 | evaluated |
| neural_factor_5 | -0.0066 | 0.0532 | 1.00 | evaluated |
| neural_factor_0 | -0.0492 | 0.2782 | 1.00 | evaluated |

## 9. Comparison With Existing Formula Factors

Formula factors (from student_laptop_report.md):
- momentum_simple_20: IC=0.0510, ICIR=0.1536
- momentum_rel_ma20: IC=0.0373, ICIR=0.1168
- momentum_simple_10: IC=0.0337, ICIR=0.1013

Neural factors (this run):
- neural_factor_6: IC=0.0455, ICIR=0.2654
- neural_factor_3: IC=0.0221, ICIR=0.2082
- neural_factor_2: IC=0.0132, ICIR=0.0851
- neural_factor_7: IC=0.0069, ICIR=0.0454
- neural_factor_1: IC=-0.0013, ICIR=0.0095

## 10. Limitations

- 样本只有 300 只股票
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

Total time: 192.29s