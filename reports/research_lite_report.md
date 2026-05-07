# Research Lite Pipeline Report (Reliability Audit)

Generated: 2026-05-08 00:23:21

## 1. Data Overview

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Stock count | 100 | 87 | OK |
| History months | 12 | 12.0 | OK |
| Date range | N/A | 20250513 to 20260507 | - |
| Total rows | N/A | 21486 | - |

## 2. Formula Factors

| Rank | Factor | RankIC | ICIR | RankIC Count |
|------|--------|--------|------|--------------|
| 1 | reversal_short_1 | 0.0374 | 0.2033 | 238 |
| 2 | reversal_short_3 | 0.0305 | 0.1643 | 238 |
| 3 | reversal_short_5 | 0.0287 | 0.1402 | 238 |
| 4 | momentum_chg_1vs60 | 0.0144 | 0.0720 | 238 |
| 5 | momentum_chg_3vs60 | 0.0143 | 0.0724 | 238 |
| 6 | momentum_chg_5vs60 | 0.0140 | 0.0703 | 238 |
| 7 | momentum_chg_1vs20 | 0.0102 | 0.0506 | 238 |
| 8 | momentum_chg_3vs20 | 0.0075 | 0.0376 | 238 |
| 9 | momentum_chg_1vs10 | 0.0059 | 0.0289 | 238 |
| 10 | momentum_chg_5vs20 | 0.0045 | 0.0233 | 238 |

- Total formula factors: 100
- Successfully evaluated: 100
- Failed: 0

## 3. Neural Factors

**No neural factors successfully evaluated (or leakage check FAIL)**

- Total neural factors: 8
- Successfully evaluated: 0

## 4. Leakage Check

| Check | Status | Details |
|-------|--------|---------|
| feature_columns | OK | All feature columns are safe |
| sequence_dates | OK | All sequences properly aligned |
| target_alignment | OK | Target properly aligned after signal_date |
| scaler_fit_scope | OK | Time periods properly separated |
| **Overall** | **OK** | - |

## 5. Future Return Alignment

- future_return calculated GROUPED BY STOCK: Yes
- MultiIndex format: Yes

## 6. Final Status

**STATUS: OK** - Pipeline successful

## 7. Limitations

- Leakage check status: OK
- Stock count target vs actual: 100 vs 87
- Date range sufficiency: OK
- Formula factors success rate: 100.0%
- This is a research prototype - NOT FOR LIVE TRADING

Total time: 115.50s