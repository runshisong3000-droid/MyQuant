# MyQuant - Quantitative Trading Framework

A simple and extensible quantitative trading framework for backtesting trading strategies.

## Features

- Data loading and preprocessing
- Strategy implementation with abstract base class
- Dual Moving Average (DualMA) strategy implementation
- Comprehensive risk metrics calculation
- Backtesting engine
- Configurable via YAML files

## Project Structure

```
MyQuant/
├── config/
│   ├── data.yaml          # Data configuration
│   └── strategy.yaml      # Strategy configuration
├── data/                  # Historical price data (CSV files)
├── src/
│   ├── __init__.py
│   ├── core/              # Core utilities
│   │   └── __init__.py
│   ├── data/              # Data loading module
│   │   ├── __init__.py
│   │   └── loader.py
│   ├── strategy/          # Strategy implementations
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── dual_ma.py
│   └── metrics/           # Risk metrics
│       ├── __init__.py
│       └── risk.py
├── tests/                 # Unit tests
│   ├── __init__.py
│   └── test_metrics.py
├── scripts/               # Run scripts
│   └── run_backtest.py
├── results/               # Backtest results
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Place your historical price data in CSV format in the `data/` directory
2. Configure `config/data.yaml` and `config/strategy.yaml`
3. Run the backtest:

```bash
python scripts/run_backtest.py
```

## Configuration

### data.yaml
- `data_source`: Data source type
- `data_dir`: Directory containing data files
- `tickers`: List of tickers to process
- `start_date`: Start date for backtesting
- `end_date`: End date for backtesting

### strategy.yaml
- `strategy`: Strategy name
- `parameters`: Strategy-specific parameters
- `initial_capital`: Initial trading capital
- `transaction_cost`: Transaction cost rate
- `slippage`: Slippage rate

## Risk Metrics

- Total Return
- Annualized Return
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Annualized Volatility
- Profit Factor
- Skewness
- Kurtosis
- VaR (Value at Risk)
- CVaR (Conditional Value at Risk)

## License

MIT License