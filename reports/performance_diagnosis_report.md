# Performance Diagnosis Report

## 1. Overview

This report analyzes the performance characteristics of the MyQuant research_medium_trial pipeline.

## 2. Data Scale

| Metric | Value |
|--------|-------|
| Stock Count | 150 |
| History Months | 18 |
| Total Rows | ~53,293 |
| Formula Factors | 108 |
| Neural Factors | 8 |

## 3. Stage Performance

### 3.1 Research Factor Pipeline

| Step | Time (s) | Notes |
|------|----------|-------|
| Data Loading | ~0.07s | From cache |
| Data Preparation | ~0.01s | Future return calculation |
| Formula Factor Generation | ~0.33s | 108 factors |
| **Formula Factor Evaluation** | **~468-546s** | **SLOWEST STEP** |
| Factor Panel Saving | ~4s | Parquet format |

### 3.2 Neural Factor Pipeline

| Step | Time (s) | Notes |
|------|----------|-------|
| Data Loading | ~0.1s | From cache |
| Data Standardization | ~0.01s | Column filtering |
| Sequence Dataset Construction | ~9-38s | Depends on lookback window |
| Time Split | ~0.04s | Simple split |
| Neural Leakage Check | ~0.03s | Validation check |
| AutoEncoder Training | ~15-28s | 5 epochs |
| Embedding Extraction | ~0.01s | Forward pass |
| Neural Factor Conversion | ~0.25s | Format conversion |
| Neural Factor Evaluation | ~9-15s | 8 factors |
| **Total** | **~34-81s** | |

### 3.3 OOS Validation Pipeline

| Step | Time (s) | Notes |
|------|----------|-------|
| Data Loading | ~1s | Multiple files |
| Feature Comparison | ~25s | Cross-validation |
| Backtest | ~0.12s | Simple backtest |
| **Total** | **~28s** | |

### 3.4 Trading Constraints Pipeline

| Step | Time (s) | Notes |
|------|----------|-------|
| Data Loading | ~1s | Price and constraint data |
| Tradable Mask Generation | ~0.03s | Filter logic |
| Constrained Backtest | ~0.13s | Simple backtest |
| **Total** | **~5s** | |

## 4. Bottleneck Analysis

### 4.1 Current Bottlenecks

| Rank | Bottleneck | Time | Percentage |
|------|------------|------|------------|
| 1 | Formula Factor Evaluation | ~468-546s | ~85-90% |
| 2 | Sequence Dataset Construction | ~9-38s | ~2-6% |
| 3 | Neural Network Training | ~15-28s | ~2-5% |

### 4.2 Root Cause Analysis

1. **Formula Factor Evaluation**
   - Each factor evaluated independently
   - Full evaluation includes: RankIC, ICIR, turnover, coverage, group analysis, decay curve, in/out sample validation
   - No parallel processing
   - No caching mechanism (until now)

2. **Sequence Dataset Construction**
   - Nested loop processing for each stock-date combination
   - Lookback window of 20 periods

## 5. Optimization Recommendations

### 5.1 Short-term (Implemented)

- ✅ **Caching**: Added `factor_eval_cache.json` to skip re-evaluation
- ✅ **Heartbeat Logging**: Output progress every 10 factors
- ✅ **Correlation Limit**: Only compute correlations for top 50 factors
- ✅ **Resume Mechanism**: `--resume --skip-completed` flags

### 5.2 Medium-term

- ⏳ Parallel evaluation using `multiprocessing`
- ⏳ Vectorized operations for groupby operations
- ⏳ Pre-compute common intermediate results

### 5.3 Long-term

- ⏳ GPU acceleration for heavy computations
- ⏳ Distributed computing for large datasets

## 6. Cloud Computing Assessment

### Current Scale vs. Cloud-Requiring Scale

| Metric | Current | Cloud Recommended |
|--------|---------|-------------------|
| Stock Count | 150 | 300-800+ |
| History | 18 months | 3-5 years |
| Factors | ~100 | 500+ |
| Models | 1 AutoEncoder | Multiple models |
| Windows | Single | Walk-forward multiple |

### Recommendation

**Current Status**: ✅ **NOT RECOMMENDED**

Current data scale is well within the capabilities of a modern laptop. The primary bottleneck is algorithmic efficiency, not computational power.

**When to Consider Cloud**:
- Stock count exceeds 300
- History exceeds 3 years
- Running walk-forward validation with >10 windows
- Parameter search with >50 combinations

## 7. Conclusion

**Current Performance Assessment**:
- ✅ Research Factor Pipeline: Functional but slow (formula evaluation)
- ✅ Neural Factor Pipeline: Acceptable performance
- ✅ OOS Validation: Fast
- ✅ Trading Constraints: Fast

**Priority Actions**:
1. Complete stage/resume implementation ✅
2. Implement factor evaluation caching ✅
3. Add performance logging ✅
4. Profile and optimize slow functions
5. Consider parallel evaluation

**Final Recommendation**:
> 当前不建议购买云算力。优先完成 stage/resume、缓存、因子评价优化。

---

*Generated: 2026-05-09*
*Profile: research_medium_trial*
*can_use_for_live_trading: false*
