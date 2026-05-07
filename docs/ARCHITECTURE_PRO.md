# MyQuant Pro - AI量化选股系统架构文档

> 版本：v1.0.0
> 创建日期：2025-05-06
> 目标：构建可展示、可迭代、可复现的专业级AI量化选股系统

---

## 📋 目录

1. [系统架构总览](#1-系统架构总览)
2. [数据层（Data Layer）](#2-数据层data-layer)
3. [因子策略层（Factor Layer）](#3-因子策略层factor-layer)
4. [AI选股模型层（Model Layer）](#4-ai选股模型层model-layer)
5. [回测层（Backtest Layer）](#5-回测层backtest-layer)
6. [日志与版本控制层（Logging Layer）](#6-日志与版本控制层logging-layer)
7. [API与展示层（API Layer）](#7-api与展示层api-layer)
8. [模块接口规范](#8-模块接口规范)
9. [Git工作流程](#9-git工作流程)
10. [开发路线图](#10-开发路线图)

---

## 1. 系统架构总览

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MyQuant Pro 系统架构                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐    │
│  │                        Layer 5: API & Display (API层)                      │    │
│  │   • REST API Server                                                      │    │
│  │   • Web Dashboard (Streamlit)                                            │    │
│  │   • Strategy Monitor                                                     │    │
│  └──────────────────────────────────────────────────────────────────────────┘    │
│                                      ↑                                          │
│  ┌──────────────────────────────────────────────────────────────────────────┐    │
│  │                    Layer 4: Logging & Version Control (日志层)            │    │
│  │   • Structured Logging (JSON)                                            │    │
│  │   • Git Integration                                                      │    │
│  │   • Experiment Tracking                                                  │    │
│  │   • Model Registry                                                       │    │
│  └──────────────────────────────────────────────────────────────────────────┘    │
│                                      ↑                                          │
│  ┌──────────────────────────────────────────────────────────────────────────┐    │
│  │                      Layer 3: Backtest Engine (回测层)                     │    │
│  │   • Event-Driven Backtest                                               │    │
│  │   • Vectorized Backtest                                                 │    │
│  │   • Transaction Simulation (Slippage, Commission)                         │    │
│  │   • Performance Analytics                                               │    │
│  └──────────────────────────────────────────────────────────────────────────┘    │
│                                      ↑                                          │
│  ┌──────────────────────────────────────────────────────────────────────────┐    │
│  │                     Layer 2: AI Model Layer (模型层)                       │    │
│  │   • XGBoost/LightGBM Models                                             │    │
│  │   • PyTorch Neural Networks                                              │    │
│  │   • Ensemble Methods                                                     │    │
│  │   • Model Training & Validation                                          │    │
│  └──────────────────────────────────────────────────────────────────────────┘    │
│                                      ↑                                          │
│  ┌──────────────────────────────────────────────────────────────────────────┐    │
│  │                    Layer 1: Factor & Strategy (因子层)                      │    │
│  │   • Technical Factors                                                   │    │
│  │   • Fundamental Factors                                                 │    │
│  │   • Style Factors                                                       │    │
│  │   • Sentiment Factors                                                   │    │
│  │   • Factor Standardization & Analysis                                    │    │
│  └──────────────────────────────────────────────────────────────────────────┘    │
│                                      ↑                                          │
│  ┌──────────────────────────────────────────────────────────────────────────┐    │
│  │                        Layer 0: Data Layer (数据层)                       │    │
│  │   • Data Sources (AkShare/Tushare/Wind)                                  │    │
│  │   • Data Cleaning & Preprocessing                                        │    │
│  │   • Feature Engineering                                                 │    │
│  │   • Data Storage (Parquet/HDF5)                                          │    │
│  └──────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 数据层（Data Layer）

### 2.1 目录结构

```
src/
├── data/
│   ├── __init__.py
│   ├── sources/
│   │   ├── __init__.py
│   │   ├── base.py              # 数据源基类
│   │   ├── akshare_source.py     # AkShare适配器
│   │   ├── tushare_source.py     # Tushare适配器
│   │   └── wind_source.py        # Wind适配器（预留）
│   │
│   ├── cleaners/
│   │   ├── __init__.py
│   │   ├── base_cleaner.py       # 清洗基类
│   │   ├── price_cleaner.py      # 价格数据清洗
│   │   ├── fundamental_cleaner.py # 财务数据清洗
│   │   └──异常值检测器
│   │
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── parquet_handler.py    # Parquet存储
│   │   └── hdf5_handler.py       # HDF5存储
│   │
│   └── loader.py                 # 统一数据加载器
```

### 2.2 数据源接口

```python
class BaseDataSource(ABC):
    """数据源基类"""

    @abstractmethod
    def get_price_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取价格数据"""
        pass

    @abstractmethod
    def get_financial_data(self, symbol: str, report_type: str) -> pd.DataFrame:
        """获取财务数据"""
        pass

    @abstractmethod
    def get_index_components(self, index_code: str) -> List[str]:
        """获取指数成分股"""
        pass
```

### 2.3 数据清洗规范

| 步骤 | 操作 | 处理方式 |
|------|------|---------|
| 1 | 缺失值检测 | 标记为NaN，不填充 |
| 2 | 异常值检测 | 3σ原则标记 |
| 3 | 停牌处理 | 保留但标记 |
| 4 | 日期对齐 | 前向填充 |
| 5 | 复权处理 | 前复权（默认） |

### 2.4 特征工程

#### 技术因子
- **趋势类**：MA5/10/20/60/120/250, EMA, MACD, DMA
- **动量类**：RSI(6/12/24), KDJ, CCI, ROC
- **波动类**：ATR, Boll(20,2), 历史波动率
- **量价类**：量比, 成交额动量, VR

#### 基本面因子
- **盈利类**：ROE, ROA, EPS, 净利润增速
- **成长类**：营收增速, 毛利率, 净利率
- **估值类**：PE, PB, PS, PCF

#### 风格因子（BARRA风格）
- Size (市值对数)
- Momentum (动量)
- Volatility (波动率)
- Growth (成长)
- Liquidity (流动性)
- Value (价值)
- Quality (质量)

#### 情绪因子
- 新闻NLP情绪分数
- 公告情感分析
- 社交媒体热度

---

## 3. 因子策略层（Factor Layer）

### 3.1 目录结构

```
src/
├── factors/
│   ├── __init__.py
│   ├── base.py                  # 因子基类
│   ├── technical/               # 技术因子
│   │   ├── momentum.py
│   │   ├── volatility.py
│   │   ├── volume.py
│   │   └── trend.py
│   ├── fundamental/            # 基本面因子
│   │   ├── profitability.py
│   │   ├── growth.py
│   │   └── valuation.py
│   ├── style/                  # 风格因子
│   │   └── barra_style.py
│   ├── sentiment/              # 情绪因子
│   │   ├── news_sentiment.py
│   │   └── social_sentiment.py
│   │
│   ├── analyzer.py             # 因子分析（IC/IR）
│   ├── standardizer.py         # 因子标准化
│   └── factor_matrix.py        # 因子矩阵构建
```

### 3.2 因子基类接口

```python
class BaseFactor(ABC):
    """因子基类"""

    def __init__(self, name: str, category: str):
        self.name = name
        self.category = category
        self.version = "1.0.0"

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值"""
        pass

    def get_metadata(self) -> dict:
        """获取因子元信息"""
        return {
            "name": self.name,
            "category": self.category,
            "version": self.version
        }
```

### 3.3 因子标准化

```python
class FactorStandardizer:
    """因子标准化处理器"""

    @staticmethod
    def zscore(factor_values: pd.Series, window: int = 252) -> pd.Series:
        """Z-Score标准化"""
        return (factor_values - factor_values.rolling(window).mean()) / \
               factor_values.rolling(window).std()

    @staticmethod
    def winsorize(factor_values: pd.Series, lower: float = 0.01,
                  upper: float = 0.99) -> pd.Series:
        """去极值（Winsorization）"""
        lower_bound = factor_values.quantile(lower)
        upper_bound = factor_values.quantile(upper)
        return factor_values.clip(lower_bound, upper_bound)

    @staticmethod
    def neutralize(factor_values: pd.Series,
                   style_factors: pd.DataFrame) -> pd.Series:
        """因子中性化（对风格因子回归残差）"""
        # 实现中性化逻辑
        pass
```

### 3.4 因子分析指标

| 指标 | 说明 | 公式 |
|------|------|------|
| IC (Information Coefficient) | 信息系数 | rank(corr(pred, actual)) |
| IR (Information Ratio) | 信息比率 | IC_mean / IC_std |
| IC_t | t统计量 | IC_mean / (IC_std / sqrt(n)) |
| IC_pvalue | p值 | t分布概率 |

---

## 4. AI选股模型层（Model Layer）

### 4.1 目录结构

```
src/
├── models/
│   ├── __init__.py
│   ├── base.py                  # 模型基类
│   ├── gradient_boosting/
│   │   ├── __init__.py
│   │   ├── xgboost_model.py
│   │   └── lightgbm_model.py
│   ├── deep_learning/
│   │   ├── __init__.py
│   │   ├── mlp_model.py
│   │   ├── lstm_model.py
│   │   └── transformer_model.py
│   ├── ensemble.py              # 模型集成
│   ├── trainer.py               # 训练器
│   ├── validator.py            # 验证器
│   └── predictor.py            # 预测器
```

### 4.2 模型基类接口

```python
class BaseModel(ABC):
    """AI模型基类"""

    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.feature_importance = None

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series,
              validation_data: Tuple[pd.DataFrame, pd.Series]) -> dict:
        """训练模型"""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        pass

    @abstractmethod
    def save(self, path: str):
        """保存模型"""
        pass

    @abstractmethod
    def load(self, path: str):
        """加载模型"""
        pass

    def get_feature_importance(self) -> pd.DataFrame:
        """获取特征重要性"""
        return self.feature_importance
```

### 4.3 XGBoost模型

```python
class XGBoostStockPicker(BaseModel):
    """基于XGBoost的选股模型"""

    def __init__(self, config: dict):
        super().__init__(config)
        self.default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }

    def train(self, X: pd.DataFrame, y: pd.Series,
              validation_data: Tuple[pd.DataFrame, pd.Series]) -> dict:

        params = {**self.default_params, **self.config}

        dtrain = xgb.DMatrix(X, label=y)
        dval = xgb.DMatrix(validation_data[0], label=validation_data[1])

        evals = [(dtrain, 'train'), (dval, 'eval')]
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=params.get('n_estimators', 100),
            evals=evals,
            early_stopping_rounds=10,
            verbose_eval=False
        )

        return self._compute_training_metrics()
```

### 4.4 模型评估指标

| 指标 | 说明 | 用途 |
|------|------|------|
| AUC | 曲线下面积 | 分类能力 |
| Accuracy | 准确率 | 整体正确率 |
| Precision | 精确率 | 预测为正的准确率 |
| Recall | 召回率 | 实际为正的预测比例 |
| F1-Score | F1分数 | 精确率和召回率的调和平均 |
| Hit Rate | 命中率 | 选出的股票中盈利比例 |

---

## 5. 回测层（Backtest Layer）

### 5.1 目录结构

```
src/
├── backtest/
│   ├── __init__.py
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── base_engine.py       # 回测引擎基类
│   │   ├── event_engine.py      # 事件驱动引擎
│   │   └── vectorized_engine.py # 向量化引擎
│   │
│   ├── simulator/
│   │   ├── __init__.py
│   │   ├── order.py            # 订单模拟
│   │   ├── position.py         # 持仓管理
│   │   ├── portfolio.py        # 组合管理
│   │   └── transaction.py       # 交易成本
│   │
│   ├── analytics/
│   │   ├── __init__.py
│   │   ├── performance.py      # 业绩归因
│   │   ├── risk_metrics.py     # 风险指标
│   │   └── attribution.py      # 归因分析
│   │
│   └── reporter/
│       ├── __init__.py
│       ├── backtest_report.py  # 回测报告
│       └── trade_journal.py    # 交易日志
```

### 5.2 回测引擎接口

```python
class BaseBacktestEngine(ABC):
    """回测引擎基类"""

    def __init__(self, config: dict):
        self.config = config
        self.portfolio = None
        self.trades = []
        self.daily_records = []

    @abstractmethod
    def run(self, strategy: BaseStrategy, data: pd.DataFrame,
            initial_capital: float = 1000000.0) -> pd.DataFrame:
        """运行回测"""
        pass

    def get_portfolio_value(self) -> pd.Series:
        """获取组合净值曲线"""
        pass

    def get_trades(self) -> pd.DataFrame:
        """获取交易记录"""
        pass

    def get_performance_summary(self) -> dict:
        """获取业绩摘要"""
        pass
```

### 5.3 交易模拟

```python
class TransactionSimulator:
    """交易成本模拟器"""

    def __init__(self, commission_rate: float = 0.001,
                 slippage_rate: float = 0.0005,
                 min_commission: float = 5.0):
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.min_commission = min_commission

    def calculate_order_cost(self, order: Order) -> dict:
        """计算订单成本"""
        base_price = order.price
        slippage = base_price * self.slippage_rate
        execution_price = base_price + slippage

        commission = max(
            order.shares * execution_price * self.commission_rate,
            self.min_commission
        )

        total_cost = order.shares * execution_price + commission

        return {
            'execution_price': execution_price,
            'commission': commission,
            'slippage': slippage * order.shares,
            'total_cost': total_cost
        }
```

### 5.4 业绩指标

| 类别 | 指标 | 说明 |
|------|------|------|
| 收益 | Total Return | 总收益率 |
| 收益 | Annual Return | 年化收益率 |
| 收益 | Alpha | 超额收益 |
| 收益 | Beta | 市场敏感度 |
| 风险 | Sharpe Ratio | 夏普比率 |
| 风险 | Sortino Ratio | 索提诺比率 |
| 风险 | Max Drawdown | 最大回撤 |
| 风险 | Calmar Ratio | 卡玛比率 |
| 风险 | VaR/CVaR | 风险价值 |
| 交易 | Win Rate | 胜率 |
| 交易 | Profit Factor | 盈亏比 |
| 交易 | Turnover | 换手率 |

---

## 6. 日志与版本控制层（Logging Layer）

### 6.1 目录结构

```
src/
├── logging/
│   ├── __init__.py
│   ├── structured_logger.py     # 结构化日志
│   ├── config.py               # 日志配置
│   └── formatters.py           # 日志格式化
│
├── versioning/
│   ├── __init__.py
│   ├── model_registry.py        # 模型注册表
│   ├── experiment_tracker.py    # 实验追踪
│   └── data_version.py         # 数据版本
```

### 6.2 结构化日志规范

```json
{
  "timestamp": "2025-05-06T10:30:00.000Z",
  "level": "INFO",
  "module": "data.loader",
  "event": "data_loaded",
  "metadata": {
    "source": "akshare",
    "symbol": "000001.SZ",
    "start_date": "2020-01-01",
    "end_date": "2025-01-01",
    "rows": 1250,
    "missing_rate": 0.02,
    "duration_seconds": 1.23
  }
}
```

### 6.3 日志事件类型

| 事件类型 | 说明 | 关键字段 |
|----------|------|---------|
| data_loaded | 数据加载 | source, symbol, rows, missing_rate |
| data_cleaned | 数据清洗 | cleaned_rows, removed_rows, outliers |
| factor_calculated | 因子计算 | factor_name, coverage, ic, ir |
| model_trained | 模型训练 | model_type, train_auc, val_auc, features |
| backtest_run | 回测运行 | strategy, start_date, end_date, sharpe |
| trade_executed | 交易执行 | symbol, action, shares, price |

### 6.4 模型注册表

```python
class ModelRegistry:
    """模型注册表"""

    def register_model(self, model_name: str, model_version: str,
                      model_path: str, metadata: dict):
        """注册模型"""
        entry = {
            "model_name": model_name,
            "version": model_version,
            "path": model_path,
            "metadata": metadata,
            "registered_at": datetime.now().isoformat(),
            "git_commit": self._get_git_commit()
        }
        self._save_to_registry(entry)

    def get_latest_model(self, model_name: str) -> dict:
        """获取最新模型"""
        pass

    def compare_models(self, model_name: str, version1: str,
                      version2: str) -> dict:
        """模型对比"""
        pass
```

---

## 7. API与展示层（API Layer）

### 7.1 目录结构

```
src/
├── api/
│   ├── __init__.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── strategies.py        # 策略API
│   │   ├── backtest.py          # 回测API
│   │   ├── models.py            # 模型API
│   │   └── data.py              # 数据API
│   │
│   └── server.py                # FastAPI服务器

dashboards/
├── app.py                        # Streamlit主应用
├── pages/
│   ├── 1_📊_Dashboard.py
│   ├── 2_🔬_Backtest.py
│   ├── 3_🤖_Models.py
│   └── 4_📈_Factors.py
```

### 7.2 API端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/strategies` | GET | 获取策略列表 |
| `/api/strategies/{id}` | POST | 创建策略 |
| `/api/backtest` | POST | 运行回测 |
| `/api/backtest/{id}` | GET | 获取回测结果 |
| `/api/models` | GET | 获取模型列表 |
| `/api/models/{id}/predict` | POST | 模型预测 |
| `/api/factors` | GET | 获取因子列表 |
| `/api/factors/{name}/values` | GET | 获取因子值 |

---

## 8. 模块接口规范

### 8.1 统一返回格式

```python
class APIResponse:
    """统一API响应格式"""

    def __init__(self, success: bool, data: Any = None,
                 error: str = None, metadata: dict = None):
        self.success = success
        self.data = data
        self.error = error
        self.metadata = metadata

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
            "timestamp": datetime.now().isoformat()
        }
```

### 8.2 配置文件规范

```yaml
# config/system.yaml
system:
  name: "MyQuant Pro"
  version: "1.0.0"
  environment: "development"  # development, production

data:
  sources:
    akshare:
      enabled: true
      cache_dir: "./data/cache"
      expire_days: 7
    tushare:
      enabled: false
      token: "${TUSHARE_TOKEN}"

logging:
  level: "INFO"
  format: "json"
  output_dir: "./logs"
  retention_days: 30

backtest:
  mode: "event_driven"
  initial_capital: 1000000.0
  commission_rate: 0.001
  slippage_rate: 0.0005
```

---

## 9. Git工作流程

### 9.1 分支策略

```
main                    # 稳定可回测版本
├── develop             # 开发分支
│   ├── feature/data-layer
│   ├── feature/factor-engineering
│   ├── feature/ml-models
│   ├── feature/backtest-engine
│   └── feature/api-dashboard
└── release/v1.0.0     # 发布版本
```

### 9.2 提交信息规范

```
<type>(<scope>): <subject>

[body]

[footer]

# 类型：
# feat: 新功能
# fix: Bug修复
# refactor: 重构
# docs: 文档
# test: 测试
# chore: 构建/工具

# 示例：
# feat(factor): 添加ROE因子计算
#
# - 实现净利润/净资产公式
# - 添加滚动窗口计算
# - 更新因子元数据
#
# Closes #123
```

### 9.3 版本标签

```
v1.0.0 - 初始版本
v1.1.0 - 添加XGBoost模型
v1.2.0 - 添加LightGBM模型
v2.0.0 - 完整系统发布
```

---

## 10. 开发路线图

### Phase 1: 基础设施 (Week 1-2)
- [ ] 项目结构搭建
- [ ] 配置管理系统
- [ ] 日志系统
- [ ] 数据源适配器

### Phase 2: 数据层 (Week 3-4)
- [ ] 数据清洗模块
- [ ] 特征工程框架
- [ ] Parquet存储
- [ ] 数据版本控制

### Phase 3: 因子层 (Week 5-6)
- [ ] 技术因子库
- [ ] 基本面因子库
- [ ] 因子标准化
- [ ] IC/IR分析

### Phase 4: 模型层 (Week 7-8)
- [ ] XGBoost模型
- [ ] LightGBM模型
- [ ] PyTorch模型
- [ ] 模型集成

### Phase 5: 回测层 (Week 9-10)
- [ ] 事件驱动引擎
- [ ] 向量化引擎
- [ ] 交易模拟
- [ ] 业绩分析

### Phase 6: API与展示 (Week 11-12)
- [ ] FastAPI服务
- [ ] Streamlit Dashboard
- [ ] 模型监控

### Phase 7: 测试与文档 (Week 13-14)
- [ ] 单元测试
- [ ] 集成测试
- [ ] 文档编写
- [ ] 演示准备

---

## 附录

### A. 技术栈

| 类别 | 技术 |
|------|------|
| 核心语言 | Python 3.9+ |
| 数据处理 | Pandas, NumPy, SciPy |
| 机器学习 | Scikit-learn, XGBoost, LightGBM, PyTorch |
| 深度学习 | PyTorch, Transformers |
| 数据存储 | Parquet, HDF5, SQLite |
| 可视化 | Matplotlib, Seaborn, Plotly, Streamlit |
| API | FastAPI, Uvicorn |
| 日志 | Structlog, Loguru |
| 版本控制 | Git, DVC |
| 测试 | Pytest, coverage |

### B. 参考项目

- [QuantConnect/Lean](https://github.com/QuantConnect/Lean) - 量化回测引擎
- [Zipline](https://github.com/quantopian/zipline) - A股回测系统
- [Qlib](https://github.com/microsoft/qlib) - 微软AI量化
- [Backtrader](https://github.com/mementum/backtrader) - Python回测
- [VectorBT](https://github.com/pwcazen/distutils-version) - 向量化回测

### C. 许可证

MIT License

---

*本文档由 MyQuant Pro 系统自动生成*
*最后更新: 2025-05-06*
