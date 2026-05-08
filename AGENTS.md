# MyQuant 项目长期上下文

## 1. 项目定位

MyQuant 是一个面向 A 股的 AI 量化选股与可视化特征学习平台。

核心方向不是普通机器学习 Demo，也不是简单预测涨跌，而是：

**真实数据 → 数据清洗 → 公式因子 → neural factors → 因子评价 → Gatekeeper 审核 → 截面排序模型 → 回测 → 可信度审计 → 可视化报告**

## 2. 当前核心目标

1. 做 A 股截面排序预测，输出 alpha score。
2. 构建自动因子研究系统。
3. 构建类计算机视觉的 Neural Feature Learning Pipeline。
4. 把神经网络 embedding 转换为 neural factors。
5. 所有因子必须经过 RankIC、ICIR、Coverage、Turnover、相关性、样本外表现和 Gatekeeper 审核。
6. 构建可视化平台，展示 AI 如何从原始量价序列学习特征。

## 3. 当前项目状态

1. **student_laptop_pipeline**：已能用 AkShare 真实数据跑通。
2. **neural_factor_pipeline**：已完成工程验证。
3. **research_lite_pipeline**：已完成可信度修复第二阶段。
4. **target_alignment**：已修复为 OK。
5. **formula factors**：已从 0/100 修复到 100/100 成功评价。
6. **测试**：59/59 全部通过。
7. **当前结果**：仍不能直接实盘，只能用于研究和展示。

## 4. 当前目录结构

| 目录 | 说明 |
|------|------|
| `src/data` | 数据获取、清洗、复权、股票池、交易日历 |
| `src/factors/auto` | 公式因子、自动因子、因子评价、筛选、审核 |
| `src/factors/neural` | 神经特征学习、sequence dataset、encoder、autoencoder、neural factors |
| `src/core` | 回测、组合、交易成本 |
| `src/metrics` | 绩效和风险指标 |
| `src/strategy` | AI 选股和 Research Agent |
| `src/validation` | 未来函数和可信度检查 |
| `scripts` | 可执行 pipeline |
| `tests` | pytest 测试 |
| `reports` | 实验报告 |
| `dashboard` | 可视化页面 |
| `docs` | 长期说明文档 |
| `prompts` | 短提示词模板 |

## 5. 开发铁律

1. **可信度 > 数据正确 > 无未来函数 > 测试通过 > 可解释报告 > 收益表现**
2. 遇到 Leakage Check、Reliability Audit、测试、数据加载、索引对齐中的 FAIL，pipeline 必须停止。
3. 不允许使用模拟数据冒充真实数据。
4. 不允许把 FAIL 改成 WARN，或把 WARN 改成 OK 来绕过问题。
5. 不允许在未通过测试时说"系统已完成"。
6. 新功能必须先分析，再写测试，再最小修改，再运行 pytest 和 pipeline。
7. 默认 CPU、本地数据、学生笔记本模式。
8. 不默认使用 GPU、云算力或 LLM API。
9. 不做实盘交易功能，除非用户明确要求。

## 6. 标准工作流

每次任务必须遵循：

1. 先分析问题
2. 输出修改计划
3. 写测试复现
4. 最小范围修复
5. 运行 pytest
6. 运行对应 pipeline
7. 输出修改文件、根因、测试结果、运行结果、剩余风险

## 7. 用户背景

用户是金融学生，懂金融、调仓、持仓、权重、净值曲线、最大回撤、Sharpe、交易成本、组合优化和风控。

但用户计算机基础较弱。

**解释代码时必须：**

1. 说明文件作用
2. 说明输入和输出
3. 说明核心类和函数
4. 说明它在量化流程中的位置
5. 说明可能出错的地方
6. 不要只堆代码

---

**更新日期：** 2026-05-08  
**版本：** v1.0