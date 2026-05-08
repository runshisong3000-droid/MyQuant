# Dashboard Data Sources

本文档描述 MyQuant Dashboard 可以读取的数据源文件。

## 1. 报告文件

### 1.1 reports/student_laptop_report.md
- **内容**: 学生笔记本模式的完整报告
- **适用页面**: Overview, Reliability Audit
- **包含**: 数据加载、因子生成、评价结果、回测摘要

### 1.2 reports/research_lite_report.md
- **内容**: 研究轻量模式报告
- **适用页面**: Overview, Factor Lab, Reliability Audit
- **包含**: 公式因子评价、神经因子评价、可信度审计

### 1.3 reports/neural_factor_report.md
- **内容**: 神经因子专项报告
- **适用页面**: Neural Feature Lab
- **包含**: 编码器训练、因子提取、评价结果

### 1.4 reports/neural_encoder_comparison.md
- **内容**: 编码器对比报告
- **适用页面**: Encoder Comparison
- **包含**: MLP/CNN/Transformer 对比、RankIC/ICIR 指标

## 2. 数据文件

### 2.1 reports/neural_factor_metadata.json
- **内容**: 神经因子元数据
- **适用页面**: Neural Feature Lab
- **包含**: 因子名称、维度、生成时间

### 2.2 data/factors/neural_factors.parquet
- **内容**: 神经因子数据
- **适用页面**: Neural Feature Lab
- **格式**: MultiIndex (date, stock)

### 2.3 data/processed/research_lite_prices.parquet
- **内容**: 处理后的价格数据
- **适用页面**: 所有页面
- **格式**: OHLCV + 复权数据

## 3. 页面数据映射

### Overview
- `reports/student_laptop_report.md`
- `reports/research_lite_report.md`

### Formula Factor Lab
- `reports/research_lite_report.md`
- `data/processed/research_lite_prices.parquet`

### Neural Feature Lab
- `reports/neural_factor_report.md`
- `reports/neural_factor_metadata.json`
- `data/factors/neural_factors.parquet`

### Encoder Comparison
- `reports/neural_encoder_comparison.md`

### Backtest Report
- `reports/student_laptop_report.md`
- `reports/research_lite_report.md`

### Reliability Audit
- `reports/research_lite_report.md`

## 4. 文件格式规范

### 报告格式
- Markdown 格式
- 支持表格、列表、标题
- 特殊标记: **WARNING**, **PASS**, **FAIL**

### 数据格式
- Parquet 格式，支持 pandas 读取
- MultiIndex 必须为 (date, stock)
- 日期格式: datetime64
- 股票代码格式: 6位数字+市场后缀 (如 000001.SZ)

## 5. 数据更新频率

- **实时**: Dashboard 启动时读取最新文件
- **按需刷新**: 用户可手动触发重新加载
- **自动检测**: 检测文件修改时间自动更新

## 6. 数据安全性

- 所有数据文件为只读
- 不修改原始数据
- 所有计算在内存中进行