import pandas as pd
import numpy as np
from typing import Dict, Any


class PerformanceMetrics:
    @staticmethod
    def calculate_cagr(portfolio_value: pd.Series) -> float:
        """计算复合年增长率"""
        years = len(portfolio_value) / 252
        total_return = portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1
        return (1 + total_return) ** (1 / years) - 1
    
    @staticmethod
    def calculate_annualized_return(portfolio_value: pd.Series) -> float:
        """计算年化收益率"""
        returns = portfolio_value.pct_change().dropna()
        return (1 + returns.mean()) ** 252 - 1
    
    @staticmethod
    def calculate_volatility(portfolio_value: pd.Series) -> float:
        """计算年化波动率"""
        returns = portfolio_value.pct_change().dropna()
        return returns.std() * np.sqrt(252)
    
    @staticmethod
    def calculate_sharpe_ratio(portfolio_value: pd.Series, risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        returns = portfolio_value.pct_change().dropna()
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    @staticmethod
    def calculate_sortino_ratio(portfolio_value: pd.Series, risk_free_rate: float = 0.02) -> float:
        """计算索提诺比率"""
        returns = portfolio_value.pct_change().dropna()
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std()
        if downside_std == 0:
            return np.nan
        return np.sqrt(252) * excess_returns.mean() / downside_std
    
    @staticmethod
    def calculate_max_drawdown(portfolio_value: pd.Series) -> float:
        """计算最大回撤"""
        peak = portfolio_value.cummax()
        drawdown = (portfolio_value - peak) / peak
        return abs(drawdown.min())
    
    @staticmethod
    def calculate_calmar_ratio(portfolio_value: pd.Series) -> float:
        """计算卡玛比率"""
        cagr = PerformanceMetrics.calculate_cagr(portfolio_value)
        max_dd = PerformanceMetrics.calculate_max_drawdown(portfolio_value)
        if max_dd == 0:
            return np.nan
        return cagr / max_dd
    
    @staticmethod
    def calculate_win_rate(trades: pd.DataFrame) -> float:
        """计算胜率"""
        if len(trades) == 0:
            return 0.0
        winning_trades = trades[trades['return'] > 0]
        return len(winning_trades) / len(trades)
    
    @staticmethod
    def calculate_profit_factor(trades: pd.DataFrame) -> float:
        """计算盈亏比"""
        if len(trades) == 0:
            return np.nan
        
        gross_profit = trades[trades['return'] > 0]['return'].sum()
        gross_loss = abs(trades[trades['return'] < 0]['return'].sum())
        
        if gross_loss == 0:
            return np.inf
        
        return gross_profit / gross_loss
    
    @staticmethod
    def evaluate(portfolio: pd.DataFrame, trades: pd.DataFrame = None) -> Dict[str, Any]:
        """综合评估"""
        portfolio_value = portfolio['total']
        
        metrics = {
            'total_return': (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1,
            'cagr': PerformanceMetrics.calculate_cagr(portfolio_value),
            'annualized_return': PerformanceMetrics.calculate_annualized_return(portfolio_value),
            'volatility': PerformanceMetrics.calculate_volatility(portfolio_value),
            'sharpe_ratio': PerformanceMetrics.calculate_sharpe_ratio(portfolio_value),
            'sortino_ratio': PerformanceMetrics.calculate_sortino_ratio(portfolio_value),
            'max_drawdown': PerformanceMetrics.calculate_max_drawdown(portfolio_value),
            'calmar_ratio': PerformanceMetrics.calculate_calmar_ratio(portfolio_value),
        }
        
        if trades is not None and len(trades) > 0:
            metrics.update({
                'number_of_trades': len(trades),
                'win_rate': PerformanceMetrics.calculate_win_rate(trades),
                'profit_factor': PerformanceMetrics.calculate_profit_factor(trades),
            })
        
        return metrics
