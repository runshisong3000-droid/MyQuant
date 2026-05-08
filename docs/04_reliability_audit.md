# 04_reliability_audit.md

## 可信度审计

### 审计项目

#### 1. 真实数据检查
- 必须使用真实市场数据
- 禁止使用模拟数据
- 数据来源必须可追溯

#### 2. 模拟数据禁止
- 不允许生成假数据
- 不允许修改真实数据
- 不允许用随机数填充

#### 3. 未来函数检查
- 检查是否使用了未来数据
- 特征只能使用 signal_date 当天及以前的数据
- future_return 必须按 stock 分组计算

#### 4. 日期分离
- **signal_date**：模型生成因子的日期
- **trade_date**：实际交易日期
- **target_date**：收益计算的目标日期
- 必须满足：signal_date < trade_date <= target_date

#### 5. target_alignment
- target_start_date 必须晚于 signal_date
- 不允许同一天预测同一天
- horizon=1 时，target_start_date 应该是下一交易日

#### 6. scaler fit scope
- 只能在训练集上拟合 scaler
- 验证集和测试集只能使用训练集的 scaler 参数
- 禁止在全数据集上拟合

#### 7. MultiIndex 对齐
- factor_data 必须是 MultiIndex(date, stock)
- future_return 必须是 MultiIndex(date, stock)
- 评价前必须检查 common index

### OK/WARN/FAIL 规则

| 状态 | 含义 | 处理 |
|------|------|------|
| OK | 通过 | 继续执行 |
| WARN | 警告 | 记录但继续 |
| FAIL | 失败 | 停止 pipeline |

### 审计流程

1. 数据加载阶段审计
2. 因子生成阶段审计
3. 因子评价阶段审计
4. 模型训练阶段审计
5. 回测阶段审计
6. 报告生成阶段审计

---

**更新日期：** 2026-05-08