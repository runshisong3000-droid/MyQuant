"""
单因子测试模块 - Factor Test

核心功能:
    - 因子IC计算
    - 因子IR计算
    - 分组收益分析
    - 换手率分析
    - 因子衰减分析
    - 因子相关性分析
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime


class FactorTest:
    """
    单因子测试
    
    评估单个因子的表现：
    1. IC分析 - 信息系数
    2. IR分析 - 信息比率
    3. 分组收益 - 多空收益、单调性
    4. 换手率 - 因子稳定性
    5. 衰减分析 - 因子时效性
    6. 相关性 - 与其他因子的相关性
    """

    def __init__(self, factor_data: pd.Series, returns: pd.Series):
        """
        初始化因子测试
        
        Args:
            factor_data: 因子数据，MultiIndex [date, stock]
            returns: 收益率数据，MultiIndex [date, stock]
        """
        self.factor_data = factor_data
        self.returns = returns
        self.dates = sorted(factor_data.index.get_level_values(0).unique())
        
    def calculate_ic(self, lookahead_periods: int = 1) -> pd.Series:
        """
        计算IC序列
        
        Args:
            lookahead_periods: 前瞻期数
            
        Returns:
            IC序列
        """
        ic_values = []
        ic_dates = []
        
        for i, date in enumerate(self.dates[:-lookahead_periods]):
            factor_vals = self.factor_data.loc[date].dropna()
            future_date = self.dates[i + lookahead_periods]
            future_returns = self.returns.loc[future_date].dropna()
            
            common_stocks = factor_vals.index.intersection(future_returns.index)
            
            if len(common_stocks) >= 10:
                ic = np.corrcoef(factor_vals.loc[common_stocks], future_returns.loc[common_stocks])[0, 1]
                if not np.isnan(ic):
                    ic_values.append(ic)
                    ic_dates.append(date)
        
        return pd.Series(ic_values, index=ic_dates)
    
    def calculate_ic_summary(self, lookahead_periods: int = 1) -> Dict[str, float]:
        """
        计算IC统计摘要
        
        Returns:
            IC统计摘要
        """
        ic_series = self.calculate_ic(lookahead_periods)
        
        return {
            'mean_ic': ic_series.mean(),
            'std_ic': ic_series.std(),
            'ir': ic_series.mean() / ic_series.std() if ic_series.std() > 0 else 0,
            'positive_ic_ratio': (ic_series > 0).mean(),
            'abs_mean_ic': np.abs(ic_series).mean(),
            'max_ic': ic_series.max(),
            'min_ic': ic_series.min(),
            'n_samples': len(ic_series)
        }
    
    def calculate_group_returns(self, n_groups: int = 10, lookahead_periods: int = 1) -> pd.DataFrame:
        """
        计算分组收益
        
        Args:
            n_groups: 分组数量
            lookahead_periods: 前瞻期数
            
        Returns:
            分组收益DataFrame
        """
        group_returns = []
        
        for i, date in enumerate(self.dates[:-lookahead_periods]):
            factor_vals = self.factor_data.loc[date].dropna()
            future_date = self.dates[i + lookahead_periods]
            future_returns = self.returns.loc[future_date].dropna()
            
            common_stocks = factor_vals.index.intersection(future_returns.index)
            
            if len(common_stocks) >= n_groups * 5:
                factor_vals = factor_vals.loc[common_stocks]
                future_returns = future_returns.loc[common_stocks]
                
                quantiles = pd.qcut(factor_vals, n_groups, labels=False)
                
                for group in range(n_groups):
                    group_mask = quantiles == group
                    group_return = future_returns[group_mask].mean()
                    
                    group_returns.append({
                        'date': date,
                        'group': group + 1,
                        'return': group_return
                    })
        
        return pd.DataFrame(group_returns)
    
    def calculate_group_summary(self, n_groups: int = 10, lookahead_periods: int = 1) -> Dict[str, Any]:
        """
        计算分组收益摘要
        
        Returns:
            分组收益摘要
        """
        group_returns = self.calculate_group_returns(n_groups, lookahead_periods)
        
        summary = {}
        
        for group in range(1, n_groups + 1):
            group_data = group_returns[group_returns['group'] == group]
            summary[f'group_{group}_mean'] = group_data['return'].mean()
            summary[f'group_{group}_std'] = group_data['return'].std()
        
        top_group = group_returns[group_returns['group'] == n_groups]
        bottom_group = group_returns[group_returns['group'] == 1]
        
        long_short = top_group['return'].values - bottom_group['return'].values
        
        summary['long_short_mean'] = long_short.mean()
        summary['long_short_std'] = long_short.std()
        summary['long_short_ir'] = long_short.mean() / long_short.std() if long_short.std() > 0 else 0
        summary['long_short_win_rate'] = (long_short > 0).mean()
        
        group_means = [summary[f'group_{i}_mean'] for i in range(1, n_groups + 1)]
        summary['monotonicity'] = np.corrcoef(range(1, n_groups + 1), group_means)[0, 1]
        
        return summary
    
    def calculate_turnover(self, n_groups: int = 10) -> pd.Series:
        """
        计算因子换手率
        
        Args:
            n_groups: 分组数量
            
        Returns:
            换手率序列
        """
        turnovers = []
        dates = []
        
        for i in range(len(self.dates) - 1):
            date_t = self.dates[i]
            date_t1 = self.dates[i + 1]
            
            factor_t = self.factor_data.loc[date_t].dropna()
            factor_t1 = self.factor_data.loc[date_t1].dropna()
            
            common_stocks = factor_t.index.intersection(factor_t1.index)
            
            if len(common_stocks) > 0:
                quantiles_t = pd.qcut(factor_t.loc[common_stocks], n_groups, labels=False)
                quantiles_t1 = pd.qcut(factor_t1.loc[common_stocks], n_groups, labels=False)
                
                turnover = (quantiles_t != quantiles_t1).mean()
                turnovers.append(turnover)
                dates.append(date_t1)
        
        return pd.Series(turnovers, index=dates)
    
    def calculate_decay(self, max_lag: int = 20) -> pd.Series:
        """
        计算因子衰减
        
        Args:
            max_lag: 最大滞后期数
            
        Returns:
            衰减曲线（IC随滞后的变化）
        """
        decay = []
        
        for lag in range(max_lag + 1):
            ic_summary = self.calculate_ic_summary(lookahead_periods=lag)
            decay.append({
                'lag': lag,
                'mean_ic': ic_summary['mean_ic'],
                'ir': ic_summary['ir']
            })
        
        return pd.DataFrame(decay).set_index('lag')
    
    def calculate_correlation(self, other_factor: pd.Series) -> float:
        """
        计算与另一个因子的相关性
        
        Args:
            other_factor: 另一个因子
            
        Returns:
            相关性系数
        """
        common_indices = self.factor_data.dropna().index.intersection(other_factor.dropna().index)
        
        if len(common_indices) > 0:
            return np.corrcoef(
                self.factor_data.loc[common_indices],
                other_factor.loc[common_indices]
            )[0, 1]
        return 0.0
    
    def run_full_test(self, n_groups: int = 10, max_lag: int = 20) -> Dict[str, Any]:
        """
        运行完整的因子测试
        
        Returns:
            完整测试结果
        """
        result = {}
        
        result['ic_summary'] = self.calculate_ic_summary()
        result['group_summary'] = self.calculate_group_summary(n_groups)
        result['turnover'] = {'mean': self.calculate_turnover(n_groups).mean()}
        result['decay'] = self.calculate_decay(max_lag).to_dict('index')
        
        return result


class MultiFactorTest:
    """
    多因子测试
    
    评估多个因子之间的关系：
    1. 因子相关性矩阵
    2. 因子分组分析
    3. 因子正交化
    """

    def __init__(self, factor_data: pd.DataFrame):
        """
        初始化多因子测试
        
        Args:
            factor_data: 因子数据，MultiIndex [date, stock]，列是因子名
        """
        self.factor_data = factor_data
    
    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """
        计算因子相关性矩阵
        
        Returns:
            相关性矩阵
        """
        return self.factor_data.corr()
    
    def calculate_ic_correlation(self, returns: pd.Series) -> pd.DataFrame:
        """
        计算IC相关性矩阵
        
        Args:
            returns: 收益率数据
            
        Returns:
            IC相关性矩阵
        """
        ic_series = {}
        
        for factor_name in self.factor_data.columns:
            factor_test = FactorTest(self.factor_data[factor_name], returns)
            ic_series[factor_name] = factor_test.calculate_ic()
        
        return pd.DataFrame(ic_series).corr()
    
    def orthogonalize_factors(self, reference_factor: str) -> pd.DataFrame:
        """
        对因子进行正交化
        
        Args:
            reference_factor: 参考因子
            
        Returns:
            正交化后的因子数据
        """
        orthogonalized = self.factor_data.copy()
        
        for factor_name in orthogonalized.columns:
            if factor_name != reference_factor:
                X = self.factor_data[reference_factor].values.reshape(-1, 1)
                y = self.factor_data[factor_name].values
                
                from sklearn.linear_model import LinearRegression
                model = LinearRegression().fit(X, y)
                residuals = y - model.predict(X)
                
                orthogonalized[factor_name] = residuals
        
        return orthogonalized
    
    def run_full_analysis(self, returns: pd.Series) -> Dict[str, Any]:
        """
        运行完整的多因子分析
        
        Returns:
            分析结果
        """
        result = {}
        
        result['correlation_matrix'] = self.calculate_correlation_matrix().to_dict()
        result['ic_correlation'] = self.calculate_ic_correlation(returns).to_dict()
        
        return result