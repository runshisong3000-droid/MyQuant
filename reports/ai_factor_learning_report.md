# AI Factor Learning Report

## 1. Purpose

本报告用于评估 AI 自动因子学习是否产生样本外增量。通过对比 formula-only、neural-only 和 formula+neural 三种特征集的样本外表现，判断神经因子是否提供了独立于传统公式因子的预测信息。

## 2. Data Source

- **数据源**: Tushare Pro API (2000 积分权限)
- **Profile**: research_medium_trial
- **数据类型**: 真实 A 股日线数据
- **数据获取方式**: 环境变量 TUSHARE_TOKEN

## 3. Data Coverage

| 指标 | 值 |
|------|-----|
| actual_stock_count | 150 |
| actual_trading_days | 356 |
| total_rows | 53,293 |
| date_range | 2024-11-15 ~ 2026-05-08 |
| can_use_for_research | true |
| can_use_for_live_trading | false |

## 4. Feature Sets

### 4.1 formula-only
- 基于技术指标的传统因子
- 因子数量: 108
- 涵盖: 动量、反转、波动率、成交量等

### 4.2 neural-only
- 通过 AutoEncoder 从量价序列学习的隐特征
- 因子数量: 8
- 输入特征: open, high, low, close, volume
- 序列长度: 20

### 4.3 formula + neural
- 组合公式因子和神经因子
- 因子数量: 116

## 5. OOS Performance

### 5.1 样本外 RankIC

| Feature Set | RankIC | ICIR | Coverage |
|-------------|--------|------|----------|
| formula-only | -0.0241 | -0.17 | 0.99 |
| neural-only | 0.0117 | 0.15 | 1.00 |
| formula+neural | -0.0250 | -0.18 | 0.99 |

### 5.2 样本外回测指标

| Feature Set | Total Return | Annual Return | Sharpe Ratio | Max Drawdown |
|-------------|--------------|---------------|--------------|--------------|
| formula-only | -0.79% | -0.54% | -1.47 | -3.2% |
| neural-only | 0.54% | 0.37% | 1.12 | -1.8% |
| formula+neural | -0.64% | -0.44% | -1.20 | -2.8% |

## 6. Incremental Value Test

### 6.1 neural-only 是否有预测力？

**初步信号**：neural-only 在样本外表现出微弱但稳定的正向 RankIC (0.0117)，Sharpe Ratio 为正 (1.12)，表明神经因子包含一定的预测信息。

### 6.2 formula + neural 是否优于 formula-only？

**否**：formula+neural 的表现略差于 formula-only（RankIC: -0.0250 vs -0.0241），表明当前神经因子未能提供增量价值，甚至可能引入噪声。

### 6.3 neural factors 是否提供增量信息？

**暂未证明**：当前实验中，神经因子未能显著提升组合表现，未能证明存在稳定增量信息。

### 6.4 neural factors 是否可能只是噪声？

**可能性存在**：虽然 neural-only 表现为正，但效果微弱，且与 formula 因子组合后反而恶化，提示可能包含噪声成分。

### 6.5 是否存在过拟合迹象？

**需要进一步验证**：neural-only 在测试集上的表现（RankIC=0.0117）相对较弱，需要更大样本和 walk-forward 验证来确认是否存在过拟合。

## 7. Trading Constraints

### 7.1 Constrained Backtest 结果

| 指标 | 值 |
|------|-----|
| Total Return | 0.17% |
| Annual Return | 0.12% |
| Sharpe Ratio | 0.00 |
| Max Drawdown | -2.67% |

### 7.2 约束影响

- 总候选数: 53,293
- 可买入: 0 (0.0%)
- 可卖出: 53,006 (99.5%)
- 过滤率: 100.0%

**注**：可买入为 0 是因为缺少 ST 标记字段，所有股票都被标记为潜在风险。

## 8. Conclusion

### 8.1 当前是否有研究价值？

**有**：research_medium_trial 成功运行，数据规模达标（150只股票，18个月），为进一步研究提供了可靠的基础。

### 8.2 当前是否不能实盘？

**是**：can_use_for_live_trading = false。当前结果仅用于研究目的，不具备实盘条件。

### 8.3 是否可以进入正式 research_medium？

**可以**：research_medium_trial 数据规模达标（actual_stock_count = 150 ≥ 100），满足进入正式 research_medium 的条件。

### 8.4 是否需要 walk-forward？

**是**：当前样本外验证仅使用单一时间窗口，需要 walk-forward 多窗口验证来确认模型稳定性。

### 8.5 是否需要改进 raw features / neural encoder？

**是**：当前 neural factors 未能证明增量价值，建议：
- 扩展原始特征维度
- 尝试不同的编码器结构（CNN、Transformer）
- 调整序列长度和隐层维度
- 改进训练策略

---

**重要声明**：本报告结果仅供研究参考，不构成任何投资建议。当前模型不具备实盘交易能力。

*Generated: 2026-05-09*
*Profile: research_medium_trial*
*can_use_for_live_trading: false*
