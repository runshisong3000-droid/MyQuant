# Research Lite Pipeline Report (Reliability Audit)

Generated: 2026-05-08 22:46:25

## 1. Data Overview

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Stock count | 150 | 5 | FAIL |
| History months | 18 | 18.0 | OK |
| Date range | N/A | 20241114 to 20260508 | - |
| Total rows | N/A | 1785 | - |

## 2. Formula Factors

**No formula factors successfully evaluated**

- Total formula factors: 102
- Successfully evaluated: 0
- Failed: 102

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

**STATUS: WARN** - Some data quality issues

## 7. Limitations

- Leakage check status: OK
- Stock count target vs actual: 150 vs 5
- Date range sufficiency: OK
- Formula factors success rate: 0.0%
- This is a research prototype - NOT FOR LIVE TRADING

Total time: 436.44s