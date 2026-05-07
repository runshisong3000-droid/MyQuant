# MyQuant 快速开始指南

## 环境设置

### 1. 激活虚拟环境

```bash
# Windows
py14venv\Scripts\activate

# Linux/Mac
source py14venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

## 快速上手

### 运行回测

```bash
python scripts/run_backtest.py
```

### 运行AI选股演示

```bash
python scripts/ai_stock_picking_demo.py
```

### 下载真实数据

```bash
python scripts/download_data.py
```

### 启动 Jupyter Notebook

```bash
jupyter lab
```

然后打开 `notebooks/01_data_exploration.ipynb` 开始探索。

## 目录说明

### 重要文件

- **README.md**: 项目概览
- **ARCHITECTURE.md**: 系统架构详解
- **QUICKSTART.md**: 快速开始 (本文档)

### 配置文件

- `config/data.yaml`: 数据源和股票池配置
- `config/strategy.yaml`: 策略参数配置
- `config/backtest.yaml`: 回测引擎配置

### 源代码

- `src/data/`: 数据加载和处理
- `src/core/`: 回测引擎核心
- `src/strategy/`: 策略实现
- `src/metrics/`: 指标计算
- `src/intelligence/`: AI模块
- `src/utils/`: 工具函数

## 开发你的第一个策略

### 1. 创建策略文件

在 `src/strategy/` 目录下创建新文件，例如 `my_strategy.py`

### 2. 继承策略基类

```python
from .base import Strategy
import pandas as pd

class MyStrategy(Strategy):
    def __init__(self, parameters=None):
        super().__init__(parameters)
    
    def generate_signals(self, data):
        # 实现你的信号生成逻辑
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        # ...
        return signals
    
    def generate_signal(self, data):
        # 单步信号生成（事件驱动用）
        return 0
    
    def get_position_size(self, price, capital):
        # 计算头寸大小
        return int(capital * 0.1 / price)
```

### 3. 配置和运行

编辑 `config/strategy.yaml` 设置你的策略参数，然后运行回测。

## 下一步

1. **学习阶段**: 阅读 `ARCHITECTURE.md` 了解系统架构
2. **数据探索**: 使用 Jupyter Notebook 探索数据
3. **策略开发**: 基于示例策略开发自己的策略
4. **AI集成**: 尝试使用 ML 模型进行选股

## 常见问题

### Q: 如何添加新的数据源？

A: 在 `src/data/sources/` 目录下创建新的源类，继承相同的接口。

### Q: 如何优化回测速度？

A: 使用向量化回测模式（`config/backtest.yaml` 中设置）。

### Q: 项目支持实盘交易吗？

A: 当前版本仅支持回测。实盘功能在路线图中，预计在阶段四推出。

## 获取帮助

- 阅读完整文档: `README.md`
- 查看架构设计: `ARCHITECTURE.md`
- 查看示例代码: `src/strategy/dual_ma.py`
