# 07_neural_feature_learning.md

## neural_feature_learning 模板

当用户说"按 neural_feature_learning 模板执行"时：

### 规则

1. **不直接新增大型模型**
   - 先评估现有模型
   - 逐步改进
   - 不盲目扩大

2. **先检查现有 neural pipeline**
   - 检查数据准备
   - 检查模型结构
   - 检查评价流程

3. **检查关键组件**
   - sequence dataset 构建
   - target alignment
   - scaler fit scope

4. **embedding 必须转成 neural factors**
   - 每个维度作为因子
   - 保留日期和股票信息
   - 创建 MultiIndex DataFrame

5. **neural factors 必须进入评价流程**
   - FactorEvaluator 评价
   - Gatekeeper 审核
   - 不跳过任何环节

6. **不允许只看 loss**
   - 必须看 RankIC
   - 必须看 ICIR
   - loss 只是中间指标

7. **必须输出关键指标**
   - RankIC
   - ICIR
   - Coverage
   - Turnover

8. **必须输出报告**
   - 记录所有结果
   - 分析因子质量
   - 给出改进建议

9. **不允许直接用于实盘**
   - 当前只是研究阶段
   - 需要更多验证
   - 不建议实盘使用

---

**适用场景：** 神经特征学习开发、审计