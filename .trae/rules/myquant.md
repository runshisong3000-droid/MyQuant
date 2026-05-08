你正在开发 MyQuant：A股AI量化选股与可视化特征学习平台。

必须遵守：

1. 目标是截面排序预测，输出 alpha score，不做简单涨跌分类。
2. 所有人工因子、公式因子、自动因子、neural factors，都必须经过 RankIC、ICIR、Coverage、Turnover、相关性、样本外表现和 Gatekeeper 审核。
3. 严禁未来函数。特征只能使用 signal_date 当天及以前数据；future_return 必须按 stock 分组计算。
4. 必须区分 signal_date、trade_date、target_date；t 日信号只能 t+1 或之后交易。
5. 遇到 Leakage Check、Reliability Audit、测试、数据加载、索引对齐中的 FAIL，pipeline 必须停止。
6. 禁止用模拟数据冒充真实数据；真实数据失败必须报错或标记 FAIL。
7. 回测必须考虑手续费、印花税、滑点、涨跌停、停牌、ST、新股、T+1；未实现必须标记 WARN。
8. 新代码必须模块化，不破坏 src/data、src/factors、src/core、src/strategy、src/validation、scripts、tests、reports。
9. 所有路径使用相对路径，保证 Windows 可运行。
10. 修改核心功能时：先分析，再写测试，再最小修改，最后运行 pytest 和 pipeline。
11. 不要隐藏异常值、跳过失败项、把 FAIL 改 WARN，或把 WARN 改 OK。
12. 默认 CPU、本地数据、学生笔记本模式；不默认使用 GPU、云算力或 LLM API。
13. 优先级：可信度 > 数据正确 > 无未来函数 > 测试通过 > 可解释报告 > 收益表现。