"""
性能指标计算模块

提供 Sharpe、Sortino、最大回撤等指标的安全计算
"""

import numpy as np
import pandas as pd


def calculate_sharpe(returns: pd.Series, annualization_factor: int = 252) -> float:
    """
    计算 Sharpe 比率

    Args:
        returns: 收益率序列
        annualization_factor: 年化因子，默认252（交易日）

    Returns:
        Sharpe 比率，如果无法计算返回 0
    """
    if returns is None or len(returns) == 0:
        return 0.0

    # 清洗数据：去除 NaN 和 inf
    cleaned = returns.dropna()
    cleaned = cleaned[np.isfinite(cleaned)]

    if len(cleaned) < 2:
        return 0.0

    mean_return = cleaned.mean()
    std_return = cleaned.std()

    if std_return == 0 or not np.isfinite(std_return):
        return 0.0

    sharpe = mean_return / std_return * np.sqrt(annualization_factor)

    if not np.isfinite(sharpe):
        return 0.0

    return float(sharpe)


def calculate_sortino(returns: pd.Series, annualization_factor: int = 252) -> float:
    """
    计算 Sortino 比率（只考虑下行风险）

    Args:
        returns: 收益率序列
        annualization_factor: 年化因子

    Returns:
        Sortino 比率，如果无法计算返回 0
    """
    if returns is None or len(returns) == 0:
        return 0.0

    cleaned = returns.dropna()
    cleaned = cleaned[np.isfinite(cleaned)]

    if len(cleaned) < 2:
        return 0.0

    mean_return = cleaned.mean()
    downside_returns = cleaned[cleaned < 0]

    if len(downside_returns) == 0:
        return 0.0

    downside_std = downside_returns.std()

    if downside_std == 0 or not np.isfinite(downside_std):
        return 0.0

    sortino = mean_return / downside_std * np.sqrt(annualization_factor)

    if not np.isfinite(sortino):
        return 0.0

    return float(sortino)


def calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
    """
    计算最大回撤

    Args:
        cumulative_returns: 累计收益率序列

    Returns:
        最大回撤（负数，表示损失比例）
    """
    if cumulative_returns is None or len(cumulative_returns) == 0:
        return 0.0

    cleaned = cumulative_returns.dropna()
    cleaned = cleaned[np.isfinite(cleaned)]

    if len(cleaned) == 0:
        return 0.0

    running_max = cleaned.cummax()
    drawdown = (cleaned - running_max) / (running_max + 1e-10)

    return float(drawdown.min())


def calculate_annual_return(cumulative_returns: pd.Series, n_days: int = None) -> float:
    """
    计算年化收益率

    Args:
        cumulative_returns: 累计收益率序列
        n_days: 回测天数，如果为 None 自动计算

    Returns:
        年化收益率
    """
    if cumulative_returns is None or len(cumulative_returns) == 0:
        return 0.0

    cleaned = cumulative_returns.dropna()
    cleaned = cleaned[np.isfinite(cleaned)]

    if len(cleaned) == 0:
        return 0.0

    total_return = cleaned.iloc[-1]
    days = n_days if n_days is not None else len(cleaned)

    if days == 0:
        return 0.0

    annual_return = (1 + total_return) ** (252 / days) - 1

    if not np.isfinite(annual_return):
        return 0.0

    return float(annual_return)


class PerformanceMetrics:
    """
    性能指标计算类（兼容旧接口）
    """

    @staticmethod
    def calculate_sharpe(returns, annualization_factor=252):
        return calculate_sharpe(returns, annualization_factor)

    @staticmethod
    def calculate_sortino(returns, annualization_factor=252):
        return calculate_sortino(returns, annualization_factor)

    @staticmethod
    def calculate_max_drawdown(cumulative_returns):
        return calculate_max_drawdown(cumulative_returns)

    @staticmethod
    def calculate_annual_return(cumulative_returns, n_days=None):
        return calculate_annual_return(cumulative_returns, n_days)
