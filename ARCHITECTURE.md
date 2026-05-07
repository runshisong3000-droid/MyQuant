# MyQuant 系统架构

## 系统概述

MyQuant 是一个四层架构的量化交易系统，支持事件驱动和向量化两种回测模式，最终目标是实现AI选股的自动化交易。

## 四层架构

```
┌─────────────────────────────────────────────────────────┐
│  Layer 1: Intelligence (智能信号层)                     │
│  ├─ 市场状态识别 (Regime Detection)                     │
│  ├─ AI选股模型 (XGBoost/LightGBM)                        │
│  └─ 特征工程 (Feature Engineering)                       │
├─────────────────────────────────────────────────────────┤
│  Layer 2: Orchestrator (编排层)                         │
│  ├─ 信号处理与组合                                     │
│  ├─ 动态权重调整                                       │
│  └─ 策略组合与参数优化                                  │
├─────────────────────────────────────────────────────────┤
│  Layer 3: Core Engine (核心引擎层)                     │
│  ├─ 事件驱动回测引擎 (Event-driven Backtest)            │
│  ├─ 向量化回测引擎 (Vectorized Backtest)                │
│  └─ 组合管理 (Portfolio Management)                     │
├─────────────────────────────────────────────────────────┤
│  Layer 4: Control & Execution (控制执行层)              │
│  ├─ 风控守卫 (Risk Management)                          │
│  ├─ 订单执行 (Order Execution)                         │
│  └─ 交易日志 (Trade Journal)                           │
└─────────────────────────────────────────────────────────┘
                        ↓
                  数据层 (Data Layer)
              ┌──────────────────────┐
              │ AkShare 数据源       │
              │ Tushare 数据源       │
              │ 本地文件缓存         │
              └──────────────────────┘
```

## 模块详细说明

### 1. 数据层 (Data Layer)

**文件位置：** `src/data/`

- **loader.py**: 统一数据加载器，支持多数据源
- **cache.py**: 本地缓存管理，减少重复下载
- **sources/**: 数据源适配器
  - akshare_source.py: AkShare 免费数据源
  - tushare_source.py: Tushare 专业数据源

**功能特性：**
- 支持多数据源（AkShare/Tushare）
- 自动本地缓存
- 统一接口设计

### 2. 核心引擎层 (Core Engine Layer)

**文件位置：** `src/core/`

- **engine.py**: 回测引擎主类
  - 事件驱动回测（Event-driven Backtest）
  - 向量化回测（Vectorized Backtest）
- **portfolio.py**: 组合管理类
  - 单资产组合
  - 多资产组合
- **event.py**: 事件处理模块

**功能特性：**
- 两种回测模式切换
- 支持佣金、滑点
- 交易记录和报告生成

### 3. 策略层 (Strategy Layer)

**文件位置：** `src/strategy/`

- **base.py**: 策略基类
- **dual_ma.py**: 双均线策略示例
- **ai_stock_picking/**: AI选股策略
  - ml_strategy.py: 机器学习选股策略

**功能特性：**
- 策略基类设计，易于扩展
- 内置技术策略
- AI选股策略框架

### 4. 指标层 (Metrics Layer)

**文件位置：** `src/metrics/`

- **risk.py**: 风险指标
  - 夏普比率、索提诺比率
  - 最大回撤、VaR、CVaR
- **performance.py**: 收益指标
  - CAGR、年化收益率
  - 胜率、盈亏比

**功能特性：**
- 完整的风险收益指标体系
- 标准化评估报告

### 5. 智能信号层 (Intelligence Layer)

**文件位置：** `src/intelligence/`

- **feature_engineer.py**: 特征工程
  - 技术指标计算
  - 自定义特征生成
- **regime_detection.py**: 市场状态识别
  - K-means聚类
  - 波动率区间检测

**功能特性：**
- 100+ 技术指标
- 市场状态自动识别

### 6. 工具层 (Utils Layer)

**文件位置：** `src/utils/`

- **visualization.py**: 可视化工具
  - Matplotlib/Seaborn 静态图
  - Plotly 交互式图表

## 配置系统

**文件位置：** `config/`

- **data.yaml**: 数据源和股票池配置
- **strategy.yaml**: 策略参数配置
- **backtest.yaml**: 回测引擎配置

## 开发路线图

### 阶段一：个人研究工具 (当前完成度：85%)
- ✅ 数据基础设施
- ✅ 回测引擎
- ✅ 基本策略
- ✅ 可视化
- 🔄 完整测试覆盖

### 阶段二：可协作研究平台 (3-6个月)
- 配置驱动策略开发
- 多策略对比框架
- 自动化报告生成
- Web界面 (Streamlit)

### 阶段三：AI辅助决策系统 (6-12个月)
- 自动特征工程
- 模型自动训练与重训练
- 市场漂移检测
- 自然语言查询界面

### 阶段四：实盘交易 (12-18个月)
- 券商接口集成
- 仿真交易
- 实时风控
- 订单执行系统

## 技术栈

- **核心语言**: Python 3.8+
- **数据处理**: Pandas, NumPy
- **机器学习**: Scikit-learn, XGBoost, LightGBM
- **数据源**: AkShare, Tushare
- **可视化**: Matplotlib, Seaborn, Plotly
- **Notebook**: Jupyter, JupyterLab

## 项目结构树

```
MyQuant/
├── config/                          # 配置文件
│   ├── data.yaml
│   ├── strategy.yaml
│   └── backtest.yaml
├── data/                            # 数据目录 (不提交)
│   └── .gitignore
├── src/
│   ├── __init__.py
│   ├── core/                        # 核心引擎
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   ├── portfolio.py
│   │   └── event.py
│   ├── data/                        # 数据层
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── cache.py
│   │   └── sources/
│   │       ├── __init__.py
│   │       ├── akshare_source.py
│   │       └── tushare_source.py
│   ├── strategy/                    # 策略层
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── dual_ma.py
│   │   └── ai_stock_picking/
│   │       └── ml_strategy.py
│   ├── metrics/                     # 指标层
│   │   ├── __init__.py
│   │   ├── risk.py
│   │   └── performance.py
│   ├── intelligence/                # 智能层
│   │   ├── __init__.py
│   │   ├── feature_engineer.py
│   │   └── regime_detection.py
│   └── utils/                       # 工具层
│       ├── __init__.py
│       └── visualization.py
├── notebooks/                       # Jupyter Notebook
│   └── 01_data_exploration.ipynb
├── tests/                           # 单元测试
│   └── __init__.py
├── scripts/                         # 可执行脚本
│   ├── run_backtest.py
│   ├── download_data.py
│   └── ai_stock_picking_demo.py
├── results/                         # 结果输出目录
├── tools/                           # 工具目录
│   └── shellcheck/
├── py14venv/                        # 虚拟环境
├── requirements.txt
├── .gitignore
├── README.md
└── ARCHITECTURE.md
```

## 设计原则

1. **模块化设计**: 每层各司其职，接口清晰
2. **可扩展性**: 易于添加新策略、数据源
3. **配置驱动**: 所有参数可通过YAML配置
4. **分层回测**: 支持事件驱动和向量化两种模式
5. **AI优先**: 为AI/ML功能预留架构空间

## 参考项目

- QuantConnect/Lean
- Backtrader
- VectorBT
- vn.py
- QuantTradingOS
