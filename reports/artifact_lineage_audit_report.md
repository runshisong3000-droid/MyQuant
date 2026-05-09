# Artifact Lineage Audit Report

## Profile: research_medium_trial
## Generated: 2026-05-09 18:51:55

## 1. Artifact Summary

| Artifact | Exists | Modified Time | Size (bytes) | Rows | Columns | Status |
|----------|--------|---------------|--------------|------|---------|--------|
| prices.parquet | True | 2026-05-09 18:07:49 | 5573834 | 53293 | 24 | FRESH |
| prices_metadata.json | True | 2026-05-09 13:55:45 | 8852 | object | 22 | STALE |
| factor_summary.parquet | True | 2026-05-09 17:05:31 | 6505 | 108 | 7 | STALE |
| factor_ic_series.parquet | True | 2026-05-09 13:57:26 | 299897 | 33142 | 3 | STALE |
| factor_correlation.parquet | True | 2026-05-09 16:45:30 | 50945 | 5670 | 3 | STALE |
| formula_factors.parquet | True | 2026-05-09 16:45:34 | 45498354 | 53293 | 110 | STALE |
| formula_factor_metadata.json | True | 2026-05-09 16:45:34 | 3212 | object | 9 | STALE |
| neural_factors.parquet | True | 2026-05-09 18:10:04 | 2471483 | 50143 | 13 | FRESH |
| neural_factor_summary.parquet | True | 2026-05-09 18:51:38 | 5215 | 8 | 7 | FRESH |
| neural_factor_metadata.json | True | 2026-05-09 18:10:04 | 880 | object | 21 | FRESH |
| oos_feature_comparison.parquet | True | 2026-05-09 18:17:26 | 8748 | 3 | 13 | FRESH |
| oos_rankic_series.parquet | True | 2026-05-09 18:17:26 | 3948 | 3 | 5 | FRESH |
| oos_backtest_summary.parquet | True | 2026-05-09 18:17:26 | 4679 | 3 | 6 | FRESH |
| oos_equity_curves.parquet | True | 2026-05-09 18:17:26 | 5290 | 201 | 3 | FRESH |
| trading_constraint_summary.parquet | True | 2026-05-09 18:17:44 | 11434 | 356 | 11 | FRESH |
| tradable_mask.parquet | True | 2026-05-09 18:17:44 | 32969 | 53293 | 12 | FRESH |
| constrained_backtest_summary.json | True | 2026-05-09 18:17:44 | 343 | object | 10 | FRESH |
| constrained_equity_curve.parquet | True | 2026-05-09 18:17:44 | 12025 | 356 | 3 | FRESH |
| constrained_drawdown_curve.parquet | True | 2026-05-09 18:17:44 | 8041 | 356 | 2 | FRESH |
| stage_status.json | True | 2026-05-09 18:17:44 | 1277 | object | 4 | FRESH |
| profile_manifest.json | True | 2026-05-09 16:45:34 | 749 | object | 10 | STALE |

## 2. Status Summary

- **FRESH (基于本轮 prices)** : 14
- **STALE (早于本轮 prices)** : 7
- **MISSING** : 0

## 3. STALE Artifacts (必须重跑)

- prices_metadata.json
- factor_summary.parquet
- factor_ic_series.parquet
- factor_correlation.parquet
- formula_factors.parquet
- formula_factor_metadata.json
- profile_manifest.json

## 5. Global File Check

- Global factor_summary.parquet: EXISTS
  - WARNING: 全局文件存在，需确认是否被误读
- Global formula_factors.parquet: EXISTS
  - WARNING: 全局文件存在，需确认是否被误读
- Global neural_factors.parquet: EXISTS
  - WARNING: 全局文件存在，需确认是否被误读

## 6. Conclusion

### Status: FAIL
- 存在 7 个 STALE artifacts，需要重跑
- 存在 0 个 MISSING artifacts
- 存在全局文件，需确认是否被误读