# 04_dashboard.md

## dashboard 模板

当用户说"按 dashboard 模板执行"时：

### 规则

1. **只读取已有文件**
   - 读取 reports/ 目录
   - 读取 parquet 文件
   - 读取 json 文件

2. **不训练模型**
   - 不运行训练
   - 不调用训练 API
   - 不生成新数据

3. **不联网**
   - 不调用外部 API
   - 不获取新数据
   - 只使用本地数据

4. **不调用 LLM API**
   - 不使用 AI 生成内容
   - 不调用外部服务
   - 纯本地操作

5. **页面结构**
   - Overview（概览）
   - Factor Lab（因子实验室）
   - Neural Feature Lab（神经特征实验室）
   - Backtest（回测）
   - Reliability Audit（可信度审计）

6. **WARN 和 FAIL 必须显示**
   - 不隐藏警告
   - 不隐藏失败
   - 明确标记问题

7. **缺文件时友好提示**
   - 提示文件缺失
   - 说明如何生成
   - 不报错崩溃

---

**适用场景：** 创建/修改可视化页面