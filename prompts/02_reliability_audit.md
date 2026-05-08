# 02_reliability_audit.md

## reliability_audit 模板

当用户说"按 reliability_audit 模板执行"时，检查：

### 审计项目

1. **是否真实数据**
   - 数据来源可追溯
   - 不是模拟数据

2. **是否模拟数据**
   - 禁止使用模拟数据
   - 禁止修改真实数据

3. **是否未来函数**
   - 特征只能使用 signal_date 及以前的数据
   - 检查是否有未来数据泄露

4. **future_return 是否按 stock 分组**
   - 使用 groupby(stock)
   - 确保正确计算

5. **signal_date / trade_date / target_date 是否分离**
   - signal_date < trade_date <= target_date
   - 不允许同一天

6. **RankIC 是否横截面计算**
   - 按日期分组计算
   - 不是时间序列相关

7. **ICIR / Sharpe 是否安全处理**
   - 处理零标准差情况
   - 处理 NaN 值

8. **FAIL 是否停止 pipeline**
   - 遇到 FAIL 必须停止
   - 不能继续执行

9. **报告是否标记 OK/WARN/FAIL**
   - 明确标记状态
   - 不隐藏问题

### 输出

- 每个检查项的状态
- 问题列表
- 建议修复方案

---

**适用场景：** 可信度审计、数据验证