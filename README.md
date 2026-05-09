# MyQuant - AI Quantitative Trading System

![GitHub stars](https://img.shields.io/github/stars/RensSong/MyQuant.svg)
![GitHub forks](https://img.shields.io/github/forks/RensSong/MyQuant.svg)
![GitHub license](https://img.shields.io/github/license/RensSong/MyQuant.svg)

An AI-powered quantitative trading and research platform for systematic investment strategies, currently being developed by an undergraduate student.


## 📋 About the Author

**Rens Song**  
- Undergraduate Student, Shanghai University  
- Shanghai, China  
- Focus: Quantitative Finance, Machine Learning, AI Trading Systems  
- LinkedIn: [Runshi Song](https://linkedin.com/in/renssong)

---

## 🌟 Features

### Core Capabilities
- **AI Stock Selection**: Machine learning-based stock picking with XGBoost/LightGBM
- **Backtesting Engine**: Event-driven and vectorized backtesting modes
- **Risk Management**: Comprehensive risk metrics and portfolio constraints
- **Data Quality**: Automated data quality assessment and reporting

### Infrastructure
- **Structured Logging**: JSON-based audit trail for all operations
- **Experiment Registry**: Complete tracking of experiments, models, and backtests
- **Version Management**: Auto-incrementing version numbers for reproducibility
- **Multi-source Data**: Abstracted data layer supporting AkShare, Tushare, and more

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MyQuant Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: Intelligence (AI Signal Layer)                        │
│  • ML Stock Picking (XGBoost/LightGBM)                          │
│  • Feature Engineering (38+ Technical Factors)                  │
│  • Market Regime Detection                                      │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: Orchestrator                                          │
│  • Signal → Portfolio → Risk Control → Execution Pipeline       │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: Core Engine                                           │
│  • Backtesting (Event-driven + Vectorized)                      │
│  • Paper Trading Simulation                                     │
├─────────────────────────────────────────────────────────────────┤
│  Layer 4: Control + Execution                                   │
│  • Risk Guardian (Stop-loss, Position Limits)                   │
│  • Trade Journal and Attribution Analysis                       │
├─────────────────────────────────────────────────────────────────┤
│  Data Layer (Data Lake)                                         │
│  • Price Data, Financials, Alternative Data                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
```bash
# Clone the repository
git clone https://github.com/RensSong/MyQuant.git
cd MyQuant

# Create virtual environment
python -m venv py14venv
py14venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Backtest
```bash
python scripts/run_backtest.py
```

### Run AI Stock Picking Demo
```bash
python scripts/ai_stock_picking_demo.py
```

### Test Logging System
```bash
python scripts/test_logging_system.py
```

---

## 📁 Project Structure

```
MyQuant/
├── config/                    # Configuration files
│   ├── data.yaml             # Data source configuration
│   ├── strategy.yaml         # Strategy parameters
│   └── backtest.yaml         # Backtest engine settings
├── src/                      # Source code
│   ├── core/                 # Core engine
│   ├── data/                 # Data layer
│   ├── strategy/             # Trading strategies
│   ├── metrics/              # Risk and performance metrics
│   ├── intelligence/         # AI and ML components
│   └── utils/                # Utilities (logging, registry, etc.)
├── scripts/                  # Executable scripts
├── tests/                    # Unit tests
├── notebooks/                # Jupyter notebooks
├── logs/                     # Log files
├── registry/                 # Experiment registries
└── artifacts/                # Generated artifacts
```

---

## 📊 Key Metrics

| Metric | Description |
|--------|-------------|
| Total Return | Overall portfolio return |
| Annual Return | Annualized return |
| Sharpe Ratio | Risk-adjusted return |
| Max Drawdown | Maximum portfolio loss |
| Win Rate | Percentage of profitable trades |
| Turnover | Portfolio turnover rate |

---

## 🛡️ Risk Management

- **Position Limits**: Single stock and sector exposure constraints
- **Stop-loss**: Automated loss limits
- **Maximum Drawdown**: Portfolio-level drawdown constraints
- **Liquidity Filters**: Volume-based trading constraints

---

## 📝 License

This project is for educational and research purposes only. Not investment advice.

---

## 📧 Contact

- Email: songrunshi@outlook.com
- GitHub: [@RunshiSong3000-droid](https://github.com/RensSong)
- Location: Shanghai, China

---

*Built with passion for quantitative finance* 📈
