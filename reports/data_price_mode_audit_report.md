# Data Price Mode Audit Report

## Profile: research_medium_trial
## Generated: 2026-05-09 18:08:00

## 1. Available Fields

| Category | Fields |
|----------|--------|
| Raw Fields | open_raw, high_raw, low_raw, close_raw, pre_close_raw |
| Adj Fields | open_adj, high_adj, low_adj, close_adj, pre_close_adj |
| Default Fields | open, high, low, close, pre_close |
| Limit Fields | up_limit, down_limit |

## 2. Price Consistency Check

### 2.1 000001.SZ Price Analysis

| Date | close_raw | up_limit | down_limit | close_adj | adj_factor |
|------|-----------|----------|------------|-----------|------------|
| 2026-05-08 | 11.30 | 12.51 | 10.23 | 1520.75 | 134.5794 |
| 2026-05-07 | 11.37 | 12.50 | 10.22 | 1530.17 | 134.5794 |
| 2026-05-06 | 11.36 | 12.64 | 10.34 | 1528.82 | 134.5794 |

### 2.2 Magnitude Analysis

- close_raw 均值: 11.54
- up_limit 均值: 12.68
- close_adj 均值: 153.72

### 2.3 Consistency Status

- close_raw 与 up_limit 数量级一致: PASS
- close_adj 与 up_limit 数量级差异过大: WARN

### 2.4 Adjustment Factor Validation

- close_adj 与 close_raw * adj_factor 一致: PASS
- 平均差异: 0.0000

## 3. Trading Constraints Usage

### 3.1 Field Usage Summary

- trading_constraint_summary.parquet 存在: OK
- 行数: 356
- 列数: 11
- 列名: date, total_candidates, tradable_count, st_filtered, suspended_filtered, limit_up_filtered, limit_down_filtered, new_stock_filtered, liquidity_filtered, capacity_filtered, filtered_count

## 4. Conclusion

### Status: FAIL
- close_adj 与 up_limit 数量级差异过大

## 5. Recommended Actions

1. 因子计算应使用 close_adj, open_adj 等复权字段
2. 交易约束应使用 close_raw, up_limit, down_limit 等原始字段
3. 涨跌停判断应基于 pct_change 或 raw 价格
4. metadata 应记录 price_mode 和字段用途