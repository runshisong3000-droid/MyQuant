# Out-of-Sample Validation Report

---

## 1. Purpose

本报告用于验证 formula factors、neural factors、formula+neural 是否有样本外信息。

---

## 2. Sample Adequacy Audit

| Metric | Value |
|--------|-------|
| Formula factor panel date range | 2024-05-20 ~ 2026-05-08 |
| Neural factor panel date range | 2024-06-17 ~ 2026-05-06 |
| Common date count | 456 |
| Common stock count | 300 |
| Test trading days | 92 |
| Test stock count | 300 |
| Sample adequacy status | **PASS** |
| Dropped rows | 279360 |
| Reason for dropped samples | Inner join alignment on (date, stock) |


---

## 3. Methodology Audit

### 3.1 Data Sources

| Data Source | Type | Status |
|-------------|------|--------|
| formula_factors.parquet | Stock-date factor panel | OK |
| neural_factors.parquet | Stock-date factor panel | OK |
| factor_summary.parquet | Factor summary table | OK |
| research_lite_prices.parquet | Price and target data | OK |

### 3.2 Feature Set Construction

| Feature Set | Source | Method |
|-------------|--------|--------|
| formula_only | formula_factors.parquet | Direct stock-date factor values |
| neural_only | neural_factors.parquet | Direct stock-date factor values |
| formula_plus_neural | Both panels | Inner join on (date, stock) |

### 3.3 Sample Alignment

- Common dates: 456
- Common stocks: 300
- Test samples per feature set: 27528

### 3.4 Methodology Issues

- OK: No issues found

### 3.5 Conclusion Reliability

**PRELIMINARY**


---

## 4. Data Split

| Period | Start Date | End Date | Trading Days |
|--------|------------|----------|--------------|
| Train | 2024-06-17 | 2025-07-29 | 273 |
| Validation | 2025-07-30 | 2025-12-11 | 91 |
| Test | 2025-12-12 | 2026-05-06 | 92 |

**Total Stocks:** 300

---

## 5. Feature Sets

| Feature Set | Feature Count | Sample Count | Description |
|-------------|---------------|--------------|-------------|
| formula_only | 135 | 27528 | 仅使用公式因子 |
| neural_only | 8 | 27526 | 仅使用神经因子 |
| formula_plus_neural | 143 | 27526 | 公式因子 + 神经因子 |

---

## 6. OOS RankIC / ICIR

| Feature Set | Test RankIC | Test ICIR | Coverage |
|-------------|-------------|-----------|----------|
| formula_only | -0.0094 | 0.0604 | 100.0% |
| neural_only | 0.0004 | 0.0040 | 100.0% |
| formula_plus_neural | -0.0096 | 0.0613 | 100.0% |

---

## 7. OOS Backtest

| Feature Set | Total Return | Annual Return | Sharpe | Max Drawdown | Turnover |
|-------------|--------------|---------------|--------|--------------|----------|
| formula_only | 0.95% | 2.62% | 1.45 | 1.31% | 0.1000 |
| neural_only | -0.00% | -0.01% | 0.00 | 1.47% | 0.1000 |
| formula_plus_neural | 1.16% | 3.20% | 1.75 | 1.17% | 0.1000 |

---

## 8. Interpretation

### 8.1 neural_only 是否有效？

0.0004 的 RankIC 表明神经因子预测能力有限。


### 8.2 formula_plus_neural 是否优于 formula_only？

formula_only RankIC: -0.0094
formula_plus_neural RankIC: -0.0096

差异: -0.0001

formula_plus_neural 未明显优于 formula_only，神经因子未提供显著增量信息。


### 8.3 是否存在过拟合迹象？

- 当前样本量较小，过拟合风险中等
- 需进一步扩大样本验证稳定性

### 8.4 当前是否可以实盘？

**否**。当前结果仅用于研究，不能直接用于实盘。

---

## 9. Limitations

- 当前样本仍有限（300 只股票，92 个测试日）
- 交易约束仍需完善（停牌、涨跌停、ST 过滤）
- Paper Trading 尚未完成
- 实盘不可用
- can_use_for_live_trading: **false**

---

**Generated At:** 2026-05-09T21:28:59.360790
**Pipeline:** oos_validation_pipeline
**Methodology Status:** PASS
**Can Use For Live Trading:** false