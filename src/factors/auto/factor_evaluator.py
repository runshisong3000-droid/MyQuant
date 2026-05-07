"""
因子评估器 - Factor Evaluator

核心功能:
    - 计算RankIC
    - 计算ICIR
    - 分组收益分析
    - 多空收益分析
    - 因子换手率
    - 因子覆盖率
    - 相关性分析
    - 衰减曲线
    - 行业中性后表现
    - 市值中性后表现
    - 样本内/样本外验证
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime


class FactorEvaluator:
    """
    因子评估器
    
    对候选因子进行全面评估：
    1. RankIC分析
    2. ICIR计算
    3. 分组收益
    4. 多空收益
    5. 换手率分析
    6. 覆盖率分析
    7. 相关性分析
    8. 衰减曲线
    9. 行业中性后表现
    10. 市值中性后表现
    11. 样本内/样本外验证
    """

    def __init__(self):
        self.leakage_keywords = [
            'future', 'target', 'label', 'return_forward', 'shift_-', 'lead_',
            'next_', 'pred_', 'forecast', 'expected', 'gt_', 'label_'
        ]
    
    def _check_future_leakage(self, factor_name: str) -> bool:
        """
        检查因子名称是否可能包含未来函数
        
        Args:
            factor_name: 因子名称
            
        Returns:
            True 如果检测到潜在未来函数风险
        """
        lower_name = factor_name.lower()
        for keyword in self.leakage_keywords:
            if keyword in lower_name:
                return True
        return False
    
    def evaluate_single(
        self, 
        factor_data: pd.Series, 
        returns: pd.Series,
        industry_data: Optional[pd.Series] = None,
        market_cap_data: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        评估单个因子
        
        Args:
            factor_data: 因子数据
            returns: 收益率数据
            industry_data: 行业数据
            market_cap_data: 市值数据
            
        Returns:
            评估结果
        """
        result = {}
        
        result['rank_ic'] = self.calculate_rank_ic(factor_data, returns)
        result['icir'] = self.calculate_icir(result['rank_ic'])
        result['turnover'] = self.calculate_turnover(factor_data)
        result['coverage'] = self.calculate_coverage(factor_data)
        
        if industry_data is not None:
            result['industry_neutral'] = self.calculate_industry_neutral_ic(
                factor_data, returns, industry_data
            )
        
        if market_cap_data is not None:
            result['market_cap_neutral'] = self.calculate_market_cap_neutral_ic(
                factor_data, returns, market_cap_data
            )
        
        return result
    
    def calculate_rank_ic(self, factor_data: pd.Series, returns: pd.Series) -> Dict[str, float]:
        """
        计算RankIC
        
        Args:
            factor_data: 因子数据
            returns: 收益率数据
            
        Returns:
            RankIC统计
        """
        if isinstance(factor_data.index, pd.MultiIndex):
            dates = factor_data.index.get_level_values(0).unique()
        else:
            dates = factor_data.index.unique()
        
        ic_values = []
        
        for date in dates:
            try:
                if isinstance(factor_data.index, pd.MultiIndex):
                    factor_vals = factor_data.loc[date]
                    return_vals = returns.loc[date]
                else:
                    factor_vals = factor_data[factor_data.index == date]
                    return_vals = returns[returns.index == date]
                
                if not isinstance(factor_vals, pd.Series):
                    continue
                if not isinstance(return_vals, pd.Series):
                    continue
                    
                factor_vals = factor_vals.dropna()
                return_vals = return_vals.dropna()
                
                common = factor_vals.index.intersection(return_vals.index)
                
                if len(common) >= 10:
                    factor_rank = factor_vals.loc[common].rank(pct=True)
                    return_rank = return_vals.loc[common].rank(pct=True)
                    
                    ic = np.corrcoef(factor_rank, return_rank)[0, 1]
                    if not np.isnan(ic):
                        ic_values.append(ic)
            except Exception as e:
                continue

        if len(ic_values) == 0:
            return {
                'mean': 0.0,
                'std': 0.0,
                'count': 0,
                'positive_ratio': 0.0,
                'timeseries': []
            }
        
        return {
            'mean': np.mean(ic_values),
            'std': np.std(ic_values),
            'count': len(ic_values),
            'positive_ratio': np.mean([ic > 0 for ic in ic_values]),
            'timeseries': ic_values
        }
    
    def calculate_icir(self, rank_ic_result: Dict[str, float]) -> float:
        """
        计算ICIR
        
        Args:
            rank_ic_result: RankIC结果
            
        Returns:
            ICIR值，如果无法计算返回0
        """
        mean_ic = rank_ic_result['mean']
        std_ic = rank_ic_result['std']
        count = rank_ic_result.get('count', 0)
        
        if count < 10:
            return 0.0
        
        if not np.isfinite(mean_ic) or not np.isfinite(std_ic):
            return 0.0
        
        if std_ic < 1e-10:
            return 0.0
        
        icir = abs(mean_ic) / std_ic
        
        if not np.isfinite(icir) or icir > 1000:
            return 0.0
        
        return icir
    
    def calculate_turnover(self, factor_data: pd.Series) -> float:
        """
        计算因子换手率
        
        Args:
            factor_data: 因子数据
            
        Returns:
            平均换手率
        """
        if isinstance(factor_data.index, pd.MultiIndex):
            dates = sorted(factor_data.index.get_level_values(0).unique())
        else:
            dates = sorted(factor_data.index.unique())
        
        turnovers = []
        
        for i in range(len(dates) - 1):
            date_t = dates[i]
            date_t1 = dates[i + 1]
            
            if isinstance(factor_data.index, pd.MultiIndex):
                try:
                    f_t = factor_data.xs(date_t, level=0).dropna()
                    f_t1 = factor_data.xs(date_t1, level=0).dropna()
                except:
                    f_t = factor_data.loc[date_t].dropna() if date_t in factor_data.index.get_level_values(0) else pd.Series()
                    f_t1 = factor_data.loc[date_t1].dropna() if date_t1 in factor_data.index.get_level_values(0) else pd.Series()
            else:
                f_t = factor_data[factor_data.index == date_t].dropna()
                f_t1 = factor_data[factor_data.index == date_t1].dropna()
            
            common = f_t.index.intersection(f_t1.index)
            
            if len(common) > 0:
                rank_t = f_t.loc[common].rank(pct=True)
                rank_t1 = f_t1.loc[common].rank(pct=True)
                
                turnover = np.mean(np.abs(rank_t - rank_t1))
                turnovers.append(turnover)
        
        return np.mean(turnovers) if turnovers else 0.0
    
    def calculate_coverage(self, factor_data: pd.Series) -> float:
        """
        计算因子覆盖率
        
        Args:
            factor_data: 因子数据
            
        Returns:
            覆盖率（非NaN比例）
        """
        return 1 - factor_data.isna().mean()
    
    def calculate_group_returns(
        self,
        factor_data: pd.Series,
        returns: pd.Series,
        n_groups: int = 10
    ) -> Dict[str, Any]:
        """
        计算分组收益
        
        Args:
            factor_data: 因子数据
            returns: 收益率数据
            n_groups: 分组数量
            
        Returns:
            分组收益结果
        """
        if isinstance(factor_data.index, pd.MultiIndex):
            dates = sorted(factor_data.index.get_level_values(0).unique())
        else:
            dates = sorted(factor_data.index.unique())
        
        group_returns = []
        
        for i in range(len(dates) - 1):
            date_t = dates[i]
            date_t1 = dates[i + 1]
            
            if isinstance(factor_data.index, pd.MultiIndex):
                f_vals = factor_data.loc[date_t].dropna()
                r_vals = returns.loc[date_t1].dropna()
            else:
                f_vals = factor_data[factor_data.index == date_t].dropna()
                r_vals = returns[returns.index == date_t1].dropna()
            
            common = f_vals.index.intersection(r_vals.index)
            
            if len(common) >= n_groups * 5:
                f_vals = f_vals.loc[common]
                r_vals = r_vals.loc[common]
                
                quantiles = pd.qcut(f_vals, n_groups, labels=False, duplicates='drop')
                
                for group in range(n_groups):
                    mask = quantiles == group
                    group_return = r_vals[mask].mean()
                    
                    group_returns.append({
                        'date': date_t1,
                        'group': group + 1,
                        'return': group_return
                    })
        
        df = pd.DataFrame(group_returns)
        
        summary = {}
        for group in range(1, n_groups + 1):
            group_data = df[df['group'] == group]
            summary[f'group_{group}_mean'] = group_data['return'].mean()
            summary[f'group_{group}_std'] = group_data['return'].std()
        
        top_group = df[df['group'] == n_groups]
        bottom_group = df[df['group'] == 1]
        
        long_short = top_group['return'].values - bottom_group['return'].values
        
        summary['long_short_mean'] = long_short.mean()
        summary['long_short_std'] = long_short.std()
        summary['long_short_ir'] = long_short.mean() / long_short.std() if long_short.std() > 0 else 0
        summary['long_short_win_rate'] = (long_short > 0).mean()
        
        group_means = [summary[f'group_{i}_mean'] for i in range(1, n_groups + 1)]
        summary['monotonicity'] = np.corrcoef(range(1, n_groups + 1), group_means)[0, 1]
        
        return summary
    
    def calculate_long_short(self, factor_data: pd.Series, returns: pd.Series) -> Dict[str, float]:
        """
        计算多空收益
        
        Args:
            factor_data: 因子数据
            returns: 收益率数据
            
        Returns:
            多空收益结果
        """
        return self.calculate_group_returns(factor_data, returns, n_groups=10)
    
    def calculate_correlation_with_existing(
        self,
        factor_data: pd.Series,
        existing_factors: Dict[str, pd.Series]
    ) -> Dict[str, float]:
        """
        计算与已有因子的相关性
        
        Args:
            factor_data: 因子数据
            existing_factors: 已有因子字典
            
        Returns:
            相关性字典
        """
        correlations = {}
        
        for name, existing_data in existing_factors.items():
            common = factor_data.dropna().index.intersection(existing_data.dropna().index)
            
            if len(common) > 0:
                corr = np.corrcoef(factor_data.loc[common], existing_data.loc[common])[0, 1]
                correlations[name] = corr
        
        return correlations
    
    def calculate_decay_curve(
        self,
        factor_data: pd.Series,
        returns: pd.Series,
        max_lag: int = 10
    ) -> pd.DataFrame:
        """
        计算衰减曲线
        
        Args:
            factor_data: 因子数据
            returns: 收益率数据
            max_lag: 最大滞后天数
            
        Returns:
            衰减曲线数据
        """
        decay_results = []
        
        for lag in range(1, max_lag + 1):
            shifted_returns = returns.groupby(level=1).shift(-lag) if isinstance(returns.index, pd.MultiIndex) else returns.shift(-lag)
            
            ic_result = self.calculate_rank_ic(factor_data, shifted_returns)
            decay_results.append({
                'lag': lag,
                'mean_ic': ic_result['mean'],
                'icir': self.calculate_icir(ic_result)
            })
        
        return pd.DataFrame(decay_results)
    
    def calculate_industry_neutral_ic(
        self,
        factor_data: pd.Series,
        returns: pd.Series,
        industry_data: pd.Series
    ) -> Dict[str, float]:
        """
        计算行业中性后的RankIC
        
        Args:
            factor_data: 因子数据
            returns: 收益率数据
            industry_data: 行业数据
            
        Returns:
            行业中性后的RankIC统计
        """
        neutral_factor = self._neutralize_by_group(factor_data, industry_data)
        return self.calculate_rank_ic(neutral_factor, returns)
    
    def calculate_market_cap_neutral_ic(
        self,
        factor_data: pd.Series,
        returns: pd.Series,
        market_cap_data: pd.Series,
        n_groups: int = 5
    ) -> Dict[str, float]:
        """
        计算市值中性后的RankIC
        
        Args:
            factor_data: 因子数据
            returns: 收益率数据
            market_cap_data: 市值数据
            n_groups: 市值分组数
            
        Returns:
            市值中性后的RankIC统计
        """
        if isinstance(market_cap_data.index, pd.MultiIndex):
            market_cap_groups = market_cap_data.groupby(level=0).apply(
                lambda x: pd.qcut(x.dropna(), n_groups, labels=False, duplicates='drop')
            )
        else:
            market_cap_groups = pd.qcut(market_cap_data.dropna(), n_groups, labels=False, duplicates='drop')
        
        neutral_factor = self._neutralize_by_group(factor_data, market_cap_groups)
        return self.calculate_rank_ic(neutral_factor, returns)
    
    def _neutralize_by_group(self, data: pd.Series, group_data: pd.Series) -> pd.Series:
        """
        按分组进行中性化（去均值）
        
        Args:
            data: 要中性化的数据
            group_data: 分组数据
            
        Returns:
            中性化后的数据
        """
        neutral_data = data.copy()
        
        if isinstance(data.index, pd.MultiIndex):
            for date in data.index.get_level_values(0).unique():
                try:
                    date_data = data.loc[date]
                    date_groups = group_data.loc[date]
                    
                    common = date_data.index.intersection(date_groups.index)
                    
                    if len(common) > 0:
                        group_means = date_data.loc[common].groupby(date_groups.loc[common]).mean()
                        
                        for stock in common:
                            group = date_groups.loc[stock]
                            if group in group_means.index:
                                neutral_data.loc[(date, stock)] = date_data.loc[stock] - group_means.loc[group]
                except:
                    pass
        else:
            group_means = data.groupby(group_data).mean()
            
            for idx in data.index:
                if idx in group_data.index and pd.notna(group_data.loc[idx]):
                    group = group_data.loc[idx]
                    if group in group_means.index:
                        neutral_data.loc[idx] = data.loc[idx] - group_means.loc[group]
        
        return neutral_data
    
    def evaluate_in_sample_out_of_sample(
        self,
        factor_data: pd.Series,
        returns: pd.Series,
        split_ratio: float = 0.7
    ) -> Dict[str, Any]:
        """
        样本内/样本外验证
        
        Args:
            factor_data: 因子数据
            returns: 收益率数据
            split_ratio: 训练集比例
            
        Returns:
            验证结果
        """
        if isinstance(factor_data.index, pd.MultiIndex):
            dates = sorted(factor_data.index.get_level_values(0).unique())
        else:
            dates = sorted(factor_data.index.unique())
        
        split_idx = int(len(dates) * split_ratio)
        train_dates = dates[:split_idx]
        test_dates = dates[split_idx:]
        
        train_mask = factor_data.index.get_level_values(0).isin(train_dates) if isinstance(factor_data.index, pd.MultiIndex) else factor_data.index.isin(train_dates)
        test_mask = factor_data.index.get_level_values(0).isin(test_dates) if isinstance(factor_data.index, pd.MultiIndex) else factor_data.index.isin(test_dates)
        
        train_factor = factor_data[train_mask]
        test_factor = factor_data[test_mask]
        
        train_returns_mask = returns.index.get_level_values(0).isin(train_dates) if isinstance(returns.index, pd.MultiIndex) else returns.index.isin(train_dates)
        test_returns_mask = returns.index.get_level_values(0).isin(test_dates) if isinstance(returns.index, pd.MultiIndex) else returns.index.isin(test_dates)
        
        train_returns = returns[train_returns_mask]
        test_returns = returns[test_returns_mask]
        
        return {
            'in_sample': {
                'rank_ic': self.calculate_rank_ic(train_factor, train_returns),
                'icir': self.calculate_icir(self.calculate_rank_ic(train_factor, train_returns))
            },
            'out_of_sample': {
                'rank_ic': self.calculate_rank_ic(test_factor, test_returns),
                'icir': self.calculate_icir(self.calculate_rank_ic(test_factor, test_returns))
            },
            'decay': self._calculate_decay(train_factor, train_returns, test_factor, test_returns)
        }
    
    def _calculate_decay(
        self,
        train_factor: pd.Series,
        train_returns: pd.Series,
        test_factor: pd.Series,
        test_returns: pd.Series
    ) -> float:
        """计算样本内外衰减"""
        train_ic = self.calculate_rank_ic(train_factor, train_returns)['mean']
        test_ic = self.calculate_rank_ic(test_factor, test_returns)['mean']
        
        if train_ic == 0:
            return 0.0
        
        return (test_ic - train_ic) / abs(train_ic)
    
    def evaluate_batch(
        self,
        candidates: List['FactorCandidate'],
        returns: pd.Series,
        existing_factors: Optional[Dict[str, pd.Series]] = None,
        industry_data: Optional[pd.Series] = None,
        market_cap_data: Optional[pd.Series] = None
    ) -> List[Dict[str, Any]]:
        """
        批量评估候选因子
        
        Args:
            candidates: 候选因子列表
            returns: 收益率数据
            existing_factors: 已有因子
            industry_data: 行业数据
            market_cap_data: 市值数据
            
        Returns:
            评估结果列表
        """
        results = []
        
        for candidate in candidates:
            result = self.evaluate_single(
                candidate.data, returns, industry_data, market_cap_data
            )
            
            if existing_factors:
                result['correlations'] = self.calculate_correlation_with_existing(
                    candidate.data, existing_factors
                )
                result['max_correlation'] = max(abs(c) for c in result['correlations'].values()) if result['correlations'] else 0.0
            
            result['group_analysis'] = self.calculate_group_returns(candidate.data, returns)
            result['decay_curve'] = self.calculate_decay_curve(candidate.data, returns)
            result['in_out_sample'] = self.evaluate_in_sample_out_of_sample(candidate.data, returns)
            
            candidate.set_evaluation_results(result)
            results.append(result)
        
        return results
