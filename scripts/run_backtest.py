import yaml
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import DataLoader
from src.strategy.dual_ma import DualMA
from src.metrics.risk import RiskMetrics


def main():
    with open('config/strategy.yaml', 'r') as f:
        strategy_config = yaml.safe_load(f)
    
    with open('config/data.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
    
    loader = DataLoader('config/data.yaml')
    
    tickers = data_config.get('tickers', [])
    if not tickers:
        print("No tickers specified in data.yaml")
        return
    
    ticker = tickers[0]
    print(f"Loading data for {ticker}...")
    
    try:
        data = loader.get_price_data(
            ticker,
            start_date=data_config.get('start_date'),
            end_date=data_config.get('end_date')
        )
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure data files exist in the data directory")
        return
    
    strategy_params = strategy_config.get('parameters', {})
    strategy = DualMA(strategy_params)
    
    initial_capital = strategy_config.get('initial_capital', 1000000.0)
    print(f"Running backtest with initial capital: {initial_capital}")
    
    portfolio = strategy.run(data, initial_capital)
    
    metrics = RiskMetrics.evaluate(portfolio)
    
    print("\n=== Backtest Results ===")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Annualized Volatility: {metrics['annualized_volatility']:.2%}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Skewness: {metrics['skewness']:.2f}")
    print(f"Kurtosis: {metrics['kurtosis']:.2f}")
    print(f"VaR (95%): {metrics['var_95']:.2%}")
    print(f"CVaR (95%): {metrics['cvar_95']:.2%}")
    
    portfolio.to_csv('results/backtest_results.csv')
    print("\nResults saved to results/backtest_results.csv")


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    main()