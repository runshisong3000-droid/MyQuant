# MyQuant Light Real Data Pipeline Report

Generated: 2026-05-07 22:46:02

## 1. Data Overview

- **Data Source**: AkShare
- **Stock Count**: 20
- **Time Range**: 20260107 ~ 20260507
- **Data Points**: 1540

## 2. Factor Research

- **Candidate Factors**: 50
- **Successfully Evaluated**: 50
- **Screened Factors**: 10

### Top 5 Factors (by RankIC)

| Factor | RankIC | ICIR | Coverage |
|--------|--------|------|----------|
| momentum_simple_20 | 0.0510 | 0.1536 | 0.99 |
| momentum_rel_ma20 | 0.0373 | 0.1168 | 0.99 |
| momentum_simple_10 | 0.0337 | 0.1013 | 0.99 |
| momentum_chg_5vs60 | 0.0322 | 0.1122 | 0.96 |
| momentum_chg_3vs60 | 0.0293 | 0.1043 | 0.96 |

### Screened Factors

- momentum_simple_20: IC=0.0510
- momentum_rel_ma20: IC=0.0373
- momentum_simple_10: IC=0.0337
- momentum_chg_5vs60: IC=0.0322
- momentum_chg_3vs60: IC=0.0293
- momentum_chg_1vs60: IC=0.0285
- volatility_downside_5: IC=-0.0301
- momentum_chg_3vs20: IC=-0.0322
- liquidity_amount_ma5: IC=-0.0325
- volatility_intraday_60: IC=-0.0362

## 3. Backtest Results

- **Backtest Period**: 2026-01-08 ~ 2026-05-07
- **Backtest Days**: 76
- **Commission**: 0.10%
- **Stamp Tax**: 0.10%
- **Slippage**: 0.10%

### Performance Metrics

| Metric | Value |
|--------|-------|
| Total Return | -6.19% |
| Annual Return | -19.09% |
| Sharpe | -1.55 |
| Max Drawdown | -0.00% |

## 4. Risk Notes

1. **Data Authenticity**: [OK] Using AkShare real data
2. **Trading Costs**: [OK] Includes commission, stamp tax, slippage
3. **Future Leakage**: [WARN] Needs manual review of factor formulas
4. **Out-of-Sample Validation**: [WARN] Needs more time validation
5. **Suspended Stock Filter**: [WARN] Not yet implemented
6. **Limit-up/down Filter**: [WARN] Not yet implemented

## 5. Reliability Audit

### Audit Checklist

| Item | Status | Details |
|------|--------|---------|
| Real Data | [OK] | Using AkShare |
| Simulated Data | [OK] | Not used |
| Future Leakage Check | [OK] | Factor names checked |
| High Risk Factors | [WARN] | None detected but manual review needed |
| Signal-Trade Lag | [OK] | t-day signal → t+1 day trade |
| RankIC Cross-Sectional | [OK] | Calculated per date |
| ICIR Safe Handling | [OK] | Returns 0 for std=0 |
| Sharpe Safe Handling | [OK] | Handles NaN/inf/zero std |
| Suspended Stock Filter | [WARN] | Not implemented |
| Limit-up/down Filter | [WARN] | Not implemented |
| Sample Period | [WARN] | Only 6 months |
| Sample Size | [WARN] | Only 20 stocks |
| Out-of-Sample | [WARN] | Not validated |

### Signal Timing

- **Signal Date**: t (end of day)
- **Trade Date**: t+1 (next trading day)
- **Return Period**: t+1 day return

### Factor Evaluation

- **Target**: Future return (t+1)
- **Method**: Spearman Rank Correlation
- **Cross-section**: Per trading day

## 6. Conclusion

Pipeline completed successfully, verified system can run stably with real data.

**Current Results Reliability**: Low-Medium (small sample, short period)

**Recommendation**: Increase sample size and extend time period before using for strategy development.

Total time: 73.01s
