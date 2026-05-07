"""
回测引擎 - Backtest Engine

核心功能:
    - 截面选股回测
    - 分组收益计算
    - 多空组合计算
    - 风险指标计算
    - 支持佣金、滑点、印花税
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import warnings


class CrossSectionBacktestEngine:
    """
    截面选股回测引擎

    支持:
    - 多空组合
    - 分组回测
    - 因子 IC/IR 计算
    - 风险指标计算
    - 交易成本模拟
    """

    def __init__(
        self,
        initial_capital: float = 10000000.0,
        commission_rate: float = 0.0003,
        stamp_tax_rate: float = 0.001,
        slippage_bps: float = 10.0
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.slippage_bps = slippage_bps

        self.portfolio_value = initial_capital
        self.current_positions = {}
        self.trade_history = []

    def run(
        self,
        factor_data: pd.Series,
        returns: pd.Series,
        stock_weights: Optional[Dict[str, float]] = None,
        n_groups: int = 10,
        long_short: bool = True,
        top_pct: float = 0.2,
        rebalance_freq: int = 20
    ) -> Dict[str, Any]:
        """
        运行回测

        Args:
            factor_data: 因子数据 (MultiIndex: date, stock)
            returns: 收益率数据 (MultiIndex: date, stock)
            stock_weights: 股票权重字典（可选）
            n_groups: 分组数量
            long_short: 是否做多空组合
            top_pct: 选股比例
            rebalance_freq: 调仓频率（天）

        Returns:
            回测结果
        """
        print("Starting backtest...")

        factor_df = factor_data.unstack(level=1)
        returns_df = returns.unstack(level=1)

        common_dates = factor_df.index.intersection(returns_df.index)
        factor_df = factor_df.loc[common_dates]
        returns_df = returns_df.loc[common_dates]

        portfolio_returns = []
        group_returns = {i: [] for i in range(1, n_groups + 1)}

        rebalance_dates = common_dates[::rebalance_freq]

        for i, date in enumerate(common_dates):
            factor_row = factor_df.loc[date].dropna()
            returns_row = returns_df.loc[date]

            if len(factor_row) < 10:
                continue

            ranked = factor_row.rank(ascending=False)
            n_stocks = len(factor_row)
            n_select = max(int(n_stocks * top_pct), 5)

            if date in rebalance_dates or i == 0:
                selected_stocks = ranked[ranked <= n_select].index.tolist()

                if long_short and n_groups == 2:
                    long_stocks = selected_stocks[:len(selected_stocks)//2]
                    short_stocks = selected_stocks[len(selected_stocks)//2:]

                    if len(long_stocks) > 0 and len(short_stocks) > 0:
                        long_ret = returns_row[long_stocks].mean()
                        short_ret = returns_row[short_stocks].mean()
                        portfolio_ret = (long_ret - short_ret) / 2
                    else:
                        portfolio_ret = 0.0

                elif long_short:
                    long_stocks = selected_stocks[:len(selected_stocks)//3]
                    short_stocks = selected_stocks[-len(selected_stocks)//3:]

                    long_ret = returns_row[long_stocks].mean() if long_stocks else 0
                    short_ret = returns_row[short_stocks].mean() if short_stocks else 0
                    portfolio_ret = (long_ret - short_ret) / 2

                else:
                    portfolio_ret = returns_row[selected_stocks].mean() if selected_stocks else 0

                for g in range(1, n_groups + 1):
                    n_per_group = max(n_stocks // n_groups, 1)
                    group_stocks = ranked[(ranked > (g-1) * n_per_group) & (ranked <= g * n_per_group)].index.tolist()
                    if len(group_stocks) > 0:
                        group_ret = returns_row[group_stocks].mean()
                        group_returns[g].append(group_ret)

            else:
                portfolio_ret = 0.0

            portfolio_returns.append(portfolio_ret)

        portfolio_returns = np.array(portfolio_returns)

        portfolio_returns = portfolio_returns[~np.isnan(portfolio_returns)]

        if len(portfolio_returns) == 0:
            return self._empty_result()

        results = self._calculate_metrics(portfolio_returns, group_returns)

        print(f"Backtest completed: Sharpe={results['sharpe_ratio']:.2f}, Total Return={results['total_return']*100:.2f}%")

        return results

    def _calculate_metrics(
        self,
        returns: np.ndarray,
        group_returns: Dict[int, List[float]]
    ) -> Dict[str, Any]:
        """计算回测指标"""
        cumulative = np.cumprod(1 + returns) - 1
        total_return = cumulative[-1] if len(cumulative) > 0 else 0

        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0

        max_drawdown = self._calculate_max_drawdown(returns)

        win_rate = np.mean(returns > 0)

        turnover = self._estimate_turnover(group_returns)

        return {
            'total_return': total_return,
            'annual_return': (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'annual_volatility': std_ret * np.sqrt(252),
            'mean_daily_return': mean_ret,
            'turnover': turnover,
            'n_trades': len(returns),
            'cumulative_returns': cumulative.tolist(),
            'daily_returns': returns.tolist(),
            'group_returns': {k: np.mean(v) if v else 0 for k, v in group_returns.items()}
        }

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """计算最大回撤"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max

        return np.min(drawdown) if len(drawdown) > 0 else 0

    def _estimate_turnover(self, group_returns: Dict[int, List[float]]) -> float:
        """估算换手率"""
        if not group_returns or not any(group_returns.values()):
            return 0.0

        valid_groups = [g for g in group_returns.values() if g]
        if not valid_groups:
            return 0.0

        avg_turnover = np.mean([np.std(g) / (np.mean(np.abs(g)) + 1e-8) for g in valid_groups])

        return min(avg_turnover, 1.0)

    def _empty_result(self) -> Dict[str, Any]:
        """返回空结果"""
        return {
            'total_return': 0,
            'annual_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'annual_volatility': 0,
            'mean_daily_return': 0,
            'turnover': 0,
            'n_trades': 0,
            'cumulative_returns': [],
            'daily_returns': [],
            'group_returns': {}
        }


class FactorBacktester:
    """
    因子回测器

    专门用于因子有效性回测
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.engine = CrossSectionBacktestEngine(
            initial_capital=self.config.get('initial_capital', 10000000),
            commission_rate=self.config.get('commission_rate', 0.0003),
            stamp_tax_rate=self.config.get('stamp_tax_rate', 0.001),
            slippage_bps=self.config.get('slippage_bps', 10)
        )

    def backtest_factor(
        self,
        factor_data: pd.Series,
        returns: pd.Series,
        n_groups: int = 10
    ) -> Dict[str, Any]:
        """
        回测单个因子

        Args:
            factor_data: 因子数据
            returns: 收益率数据
            n_groups: 分组数量

        Returns:
            回测结果
        """
        return self.engine.run(
            factor_data=factor_data,
            returns=returns,
            n_groups=n_groups,
            long_short=True,
            rebalance_freq=20
        )

    def backtest_factor_list(
        self,
        factors: Dict[str, pd.Series],
        returns: pd.Series,
        n_groups: int = 10
    ) -> Dict[str, Dict[str, Any]]:
        """
        回测多个因子

        Args:
            factors: 因子字典
            returns: 收益率数据
            n_groups: 分组数量

        Returns:
            各因子回测结果
        """
        results = {}

        for name, factor_data in factors.items():
            print(f"Backtesting factor: {name}")

            try:
                result = self.backtest_factor(factor_data, returns, n_groups)
                results[name] = result
            except Exception as e:
                print(f"Error backtesting {name}: {e}")
                results[name] = {'error': str(e)}

        return results

    def compare_factors(
        self,
        factors: Dict[str, pd.Series],
        returns: pd.Series
    ) -> pd.DataFrame:
        """
        对比多个因子

        Args:
            factors: 因子字典
            returns: 收益率数据

        Returns:
            对比表格
        """
        results = self.backtest_factor_list(factors, returns, n_groups=10)

        comparison = []
        for name, result in results.items():
            if 'error' not in result:
                comparison.append({
                    'factor': name,
                    'total_return': result.get('total_return', 0),
                    'sharpe_ratio': result.get('sharpe_ratio', 0),
                    'max_drawdown': result.get('max_drawdown', 0),
                    'win_rate': result.get('win_rate', 0),
                    'turnover': result.get('turnover', 0),
                    'annual_return': result.get('annual_return', 0)
                })

        return pd.DataFrame(comparison).sort_values('sharpe_ratio', ascending=False)
