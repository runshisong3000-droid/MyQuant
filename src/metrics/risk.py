import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any


class RiskMetrics:
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    @staticmethod
    def calculate_max_drawdown(portfolio: pd.DataFrame) -> float:
        portfolio['cumulative_return'] = (portfolio['total'] / portfolio['total'].iloc[0]) - 1
        portfolio['peak'] = portfolio['cumulative_return'].cummax()
        portfolio['drawdown'] = portfolio['cumulative_return'] - portfolio['peak']
        max_drawdown = portfolio['drawdown'].min()
        return abs(max_drawdown)
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std()
        if downside_std == 0:
            return np.nan
        return np.sqrt(252) * excess_returns.mean() / downside_std
    
    @staticmethod
    def calculate_volatility(returns: pd.Series) -> float:
        return returns.std() * np.sqrt(252)
    
    @staticmethod
    def calculate_beta(portfolio_returns: pd.Series, market_returns: pd.Series) -> float:
        covariance = np.cov(portfolio_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        if market_variance == 0:
            return np.nan
        return covariance / market_variance
    
    @staticmethod
    def calculate_alpha(portfolio_returns: pd.Series, market_returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        beta = RiskMetrics.calculate_beta(portfolio_returns, market_returns)
        if np.isnan(beta):
            return np.nan
        portfolio_excess = portfolio_returns.mean() * 252 - risk_free_rate
        market_excess = market_returns.mean() * 252 - risk_free_rate
        return portfolio_excess - beta * market_excess
    
    @staticmethod
    def calculate_win_rate(signals: pd.DataFrame) -> float:
        trades = signals[signals['signal'] != 0]
        if len(trades) == 0:
            return 0.0
        winning_trades = len(trades[trades['signal'] > 0])
        return winning_trades / len(trades)
    
    @staticmethod
    def calculate_profit_factor(portfolio: pd.DataFrame) -> float:
        returns = portfolio['total'].pct_change().dropna()
        profits = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        if losses == 0:
            return np.inf
        return profits / losses
    
    @staticmethod
    def evaluate(portfolio: pd.DataFrame) -> Dict[str, Any]:
        returns = portfolio['total'].pct_change().dropna()
        
        metrics = {
            'total_return': (portfolio['total'].iloc[-1] / portfolio['total'].iloc[0]) - 1,
            'annualized_return': (1 + returns.mean()) ** 252 - 1,
            'sharpe_ratio': RiskMetrics.calculate_sharpe_ratio(returns),
            'sortino_ratio': RiskMetrics.calculate_sortino_ratio(returns),
            'max_drawdown': RiskMetrics.calculate_max_drawdown(portfolio),
            'annualized_volatility': RiskMetrics.calculate_volatility(returns),
            'profit_factor': RiskMetrics.calculate_profit_factor(portfolio),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'var_95': np.percentile(returns, 5),
            'cvar_95': returns[returns <= np.percentile(returns, 5)].mean()
        }
        
        return metrics