# Trading Constraints Report

Generated: 2026-05-09 21:29:22

---

## 1. Purpose

本报告用于检查 A 股真实交易约束对回测结果的影响。

---

## 2. Data Availability

| Field | Status | Notes |
|-------|--------|-------|
| ST 标记 | WARN | WARN - 缺少 is_st 或 stock_name 字段 |
| 停牌标记 | APPROXIMATE | APPROXIMATE - 使用成交量/成交额为0近似判断 |
| 涨跌停 | APPROXIMATE | APPROXIMATE - 使用涨跌幅近似判断 |
| 上市日期 | WARN | WARN - 缺少上市日期字段 |
| 成交额 | OK | OK |

---

## 3. Constraint Rules

| Constraint | Enabled | Parameter |
|------------|---------|-----------|
| ST Filter | True | - |
| Suspended Filter | True | volume/amount == 0 |
| Limit Up/Down Filter | True | 10.0% |
| New Stock Filter | True | Min 60 days |
| Liquidity Filter | True | Min 50,000,000 |
| Capacity Filter | True | Max 5.0% |

---

## 4. Constraint Summary

### 4.1 Daily Statistics

- Total dates: 477
- Average daily candidates: 299
- Average daily tradable: 0
- Average daily filtered: 305

### 4.2 Filter Reasons (Total)

| Filter Reason | Count |
|---------------|-------|
| ST 股票 | 0 |
| 停牌 | 0 |
| 涨停 | 2663 |
| 跌停 | 871 |
| 新股 | 0 |
| 流动性不足 | 142830 |
| 容量不足 | 0 |

---

## 5. Backtest Impact

### 5.1 Constrained Backtest Results

| Metric | Value |
|--------|-------|
| Total Return | 0.47% |
| Annual Return | 0.25% |
| Sharpe Ratio | 0.01 |
| Max Drawdown | -2.67% |
| Turnover | 0.2500 |

### 5.2 Unconstrained vs Constrained

| Metric | Unconstrained | Constrained | Change |
|--------|---------------|-------------|--------|
| Can Use For Live Trading | false | false | - |

---

## 6. Limitations

- **ST 数据缺失**: 当前数据源没有 ST 标记字段，无法精确过滤 ST 股票。
- **涨跌停近似**: 使用涨跌幅近似判断涨跌停，可能存在误差。
- **停牌近似**: 使用成交量/成交额为0判断停牌，可能存在误判。
- **新股数据缺失**: 当前数据源没有上市日期字段，无法精确过滤新股。
- **非实盘**: 当前只是更真实的研究回测，不是实盘交易系统。

---

## 7. Conclusion

1. 交易约束模块已建立，支持 ST、停牌、涨跌停、新股、流动性、容量约束。
2. 部分约束使用近似判断，需注意数据局限性。
3. 当前回测已考虑真实 A 股交易约束，更接近实际交易环境。
4. **本系统仍不能用于实盘交易** (can_use_for_live_trading: false)。

---

**Generated At:** 2026-05-09T21:29:22.304820
**Pipeline:** trading_constraints_pipeline
**Can Use For Live Trading:** false