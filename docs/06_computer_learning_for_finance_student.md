# 06_computer_learning_for_finance_student.md

## 给金融学生的编程入门

### 基本概念

#### Python 文件

- `.py` 文件是 Python 代码文件
- 每个文件包含一组相关的函数和类
- 可以被其他文件引用

#### Function（函数）

- 一段可重用的代码
- 接受输入，返回输出
- 例子：计算 RankIC 的函数

#### Class（类）

- 数据和方法的集合
- 可以创建多个实例
- 例子：FactorEvaluator 类

#### Import（导入）

- 使用其他文件的代码
- `import pandas as pd` 导入 pandas 库
- `from src.factors.auto import FactorEvaluator` 导入特定类

### 数据结构

#### DataFrame

- 表格形式的数据
- 类似 Excel 表格
- 有行和列
- 可以进行筛选、计算、合并等操作

#### MultiIndex

- 多层索引
- 在量化中通常是 (date, stock)
- 方便按日期和股票进行分组操作

### 开发工具

#### Pipeline

- 一系列步骤的组合
- 数据依次经过每个步骤
- 输出最终结果

#### pytest

- Python 的测试框架
- 自动运行测试用例
- 验证代码正确性

#### parquet

- 高效的数据存储格式
- 比 CSV 更快、更小
- 适合存储大量金融数据

#### Dashboard

- 可视化界面
- 展示数据和结果
- 方便查看和分析

### 学习建议

1. 从简单的 pipeline 开始理解
2. 运行测试看输出
3. 修改代码观察变化
4. 逐步增加复杂度

---

**更新日期：** 2026-05-08