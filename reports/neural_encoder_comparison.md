# Neural Encoder Comparison Report

Generated: 2026-05-08 15:49:08

## 1. Purpose

对比 MLP、CNN1D、TinyTransformer 三种编码器的表现。

## 2. Comparison Metrics

| Encoder | Train Loss | Val Loss | Time (s) | Avg RankIC | Best RankIC | Avg ICIR | Best ICIR | Passing Factors |
|---------|------------|----------|----------|------------|-------------|----------|-----------|-----------------|
| MLP | 0.032824 | 0.045647 | 19.91 | 0.0082 | 0.0294 | 0.0995 | 0.2282 | 2 |
| CNN | 0.050602 | 0.069817 | 13.27 | -0.0028 | 0.0216 | 0.0591 | 0.1605 | 1 |
| TRANSFORMER | 0.051509 | 0.072014 | 29.97 | 0.0011 | 0.0141 | 0.0617 | 0.0861 | 0 |

## 3. Detailed Results by Encoder

### MLP

- **Final Train Loss**: 0.032824
- **Final Val Loss**: 0.045647
- **Training Time**: 19.91s
- **Passing Factors**: 2

| Factor | RankIC | ICIR | Coverage |
|--------|--------|------|----------|
| neural_factor_2 | 0.0294 | 0.2282 | 1.00 |
| neural_factor_6 | 0.0268 | 0.2123 | 1.00 |
| neural_factor_5 | 0.0106 | 0.0628 | 1.00 |
| neural_factor_1 | 0.0104 | 0.0738 | 1.00 |
| neural_factor_3 | 0.0089 | 0.0626 | 1.00 |
| neural_factor_4 | 0.0026 | 0.0165 | 1.00 |
| neural_factor_7 | -0.0092 | 0.0582 | 1.00 |
| neural_factor_0 | -0.0135 | 0.0819 | 1.00 |

### CNN

- **Final Train Loss**: 0.050602
- **Final Val Loss**: 0.069817
- **Training Time**: 13.27s
- **Passing Factors**: 1

| Factor | RankIC | ICIR | Coverage |
|--------|--------|------|----------|
| neural_factor_2 | 0.0216 | 0.1605 | 1.00 |
| neural_factor_7 | 0.0033 | 0.0188 | 1.00 |
| neural_factor_3 | -0.0009 | 0.0054 | 1.00 |
| neural_factor_5 | -0.0025 | 0.0140 | 1.00 |
| neural_factor_1 | -0.0045 | 0.0281 | 1.00 |
| neural_factor_6 | -0.0120 | 0.0746 | 1.00 |
| neural_factor_4 | -0.0120 | 0.0729 | 1.00 |
| neural_factor_0 | -0.0158 | 0.0980 | 1.00 |

### TRANSFORMER

- **Final Train Loss**: 0.051509
- **Final Val Loss**: 0.072014
- **Training Time**: 29.97s
- **Passing Factors**: 0

| Factor | RankIC | ICIR | Coverage |
|--------|--------|------|----------|
| neural_factor_6 | 0.0141 | 0.0852 | 1.00 |
| neural_factor_0 | 0.0140 | 0.0861 | 1.00 |
| neural_factor_7 | 0.0104 | 0.0640 | 1.00 |
| neural_factor_5 | 0.0070 | 0.0420 | 1.00 |
| neural_factor_2 | -0.0087 | 0.0498 | 1.00 |
| neural_factor_3 | -0.0089 | 0.0515 | 1.00 |
| neural_factor_4 | -0.0093 | 0.0568 | 1.00 |
| neural_factor_1 | -0.0095 | 0.0582 | 1.00 |

## 4. Leakage Check

| Check | Status |
|-------|--------|
| feature_columns | OK |
| sequence_dates | OK |
| target_alignment | OK |
| scaler_fit_scope | OK |

## 5. Conclusion

- **Best Encoder by Avg RankIC**: MLP
- **Best Avg RankIC**: 0.0082
- **Leakage Check**: PASS

## 6. Important Notes

- 当前只用于研究，**不能直接实盘**
- 所有 neural factors 都经过 RankIC/ICIR/Coverage 评价
- 通过可信度审计和未来函数检查
- 样本量有限，结果仅供参考

Total time: 93.45s