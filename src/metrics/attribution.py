"""
归因分析模块 - Attribution Analysis

核心功能:
    - 绩效归因（Performance Attribution）
    - 因子归因（Factor Attribution）
    - Brinson归因
    - 风险归因
    - 时序归因

这是量化策略分析的核心工具，用于理解收益来源。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime


class PerformanceAttribution:
    """
    绩效归因基类
    
    归因分析的核心概念:
    1. 收益归因 - 分析收益来源
    2. 风险归因 - 分析风险来源
    3. 时序归因 - 时间维度的归因
    """

    def __init__(self):
        pass
    
    def analyze(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, Any]:
        """执行归因分析"""
        raise NotImplementedError


class BrinsonAttribution(PerformanceAttribution):
    """
    Brinson归因模型
    
    将组合收益分解为:
    1. 配置效应（Allocation Effect）- 行业选择
    2. 选股效应（Selection Effect）- 个股选择
    3. 交互效应（Interaction Effect）- 配置与选股的交互
    
    公式:
    R_p - R_b = Σ(w_p,i - w_b,i) * R_b,i + Σw_p,i * (R_p,i - R_b,i) + Σ(w_p,i - w_b,i) * (R_p,i - R_b,i)
    """

    def __init__(self):
        super().__init__()
    
    def analyze(
        self,
        portfolio_weights: pd.DataFrame,
        portfolio_returns: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        benchmark_returns: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        执行Brinson归因分析
        
        Args:
            portfolio_weights: 组合权重 [日期, 行业/因子]
            portfolio_returns: 组合收益 [日期, 行业/因子]
            benchmark_weights: 基准权重 [日期, 行业/因子]
            benchmark_returns: 基准收益 [日期, 行业/因子]
            
        Returns:
            归因结果
        """
        dates = portfolio_weights.index
        attribution_results = []
        
        for date in dates:
            pw = portfolio_weights.loc[date]
            pr = portfolio_returns.loc[date]
            bw = benchmark_weights.loc[date]
            br = benchmark_returns.loc[date]
            
            allocation = (pw - bw) * br
            selection = pw * (pr - br)
            interaction = (pw - bw) * (pr - br)
            
            attribution_results.append({
                'date': date,
                'allocation': allocation.sum(),
                'selection': selection.sum(),
                'interaction': interaction.sum(),
                'total': allocation.sum() + selection.sum() + interaction.sum()
            })
        
        attribution_df = pd.DataFrame(attribution_results).set_index('date')
        
        summary = {
            'total_allocation': attribution_df['allocation'].sum(),
            'total_selection': attribution_df['selection'].sum(),
            'total_interaction': attribution_df['interaction'].sum(),
            'total_excess_return': attribution_df['total'].sum(),
            'attribution_df': attribution_df,
            'component_analysis': self._analyze_components(attribution_df)
        }
        
        return summary
    
    def _analyze_components(self, attribution_df: pd.DataFrame) -> Dict[str, float]:
        """分析归因组成"""
        total = attribution_df['total'].sum()
        
        return {
            'allocation_contribution': attribution_df['allocation'].sum() / total if total != 0 else 0,
            'selection_contribution': attribution_df['selection'].sum() / total if total != 0 else 0,
            'interaction_contribution': attribution_df['interaction'].sum() / total if total != 0 else 0
        }


class FactorAttribution(PerformanceAttribution):
    """
    因子归因模型
    
    将组合收益分解为各个因子的贡献:
    - BARRA风格因子
    - 行业因子
    - 特质因子
    
    模型: R_p = Σ(beta_i * factor_return_i) + epsilon
    """

    def __init__(self, factor_returns: pd.DataFrame, factor_exposures: pd.DataFrame):
        super().__init__()
        self.factor_returns = factor_returns
        self.factor_exposures = factor_exposures
    
    def analyze(self, portfolio_returns: pd.Series) -> Dict[str, Any]:
        """
        执行因子归因分析
        
        Args:
            portfolio_returns: 组合收益序列
            
        Returns:
            因子归因结果
        """
        factor_contributions = []
        
        for date in portfolio_returns.index:
            if date in self.factor_returns.index and date in self.factor_exposures.index:
                fr = self.factor_returns.loc[date]
                fe = self.factor_exposures.loc[date]
                
                contributions = fr * fe
                total_factor_return = contributions.sum()
                specific_return = portfolio_returns.loc[date] - total_factor_return
                
                factor_contributions.append({
                    'date': date,
                    **contributions.to_dict(),
                    'total_factor_return': total_factor_return,
                    'specific_return': specific_return,
                    'actual_return': portfolio_returns.loc[date]
                })
        
        contributions_df = pd.DataFrame(factor_contributions).set_index('date')
        
        summary = {
            'factor_contributions': contributions_df,
            'factor_importance': self._calculate_factor_importance(contributions_df),
            'explained_variance': self._calculate_explained_variance(contributions_df, portfolio_returns),
            'cumulative_attribution': contributions_df.cumsum()
        }
        
        return summary
    
    def _calculate_factor_importance(self, contributions_df: pd.DataFrame) -> pd.Series:
        """计算因子重要性"""
        factor_cols = [col for col in contributions_df.columns 
                       if col not in ['total_factor_return', 'specific_return', 'actual_return']]
        
        return contributions_df[factor_cols].abs().sum().sort_values(ascending=False)
    
    def _calculate_explained_variance(self, contributions_df: pd.DataFrame, actual_returns: pd.Series) -> float:
        """计算解释方差"""
        predicted = contributions_df['total_factor_return']
        actual = actual_returns.loc[predicted.index]
        
        ss_total = np.sum((actual - actual.mean()) ** 2)
        ss_residual = np.sum((actual - predicted) ** 2)
        
        return 1 - (ss_residual / ss_total) if ss_total != 0 else 0


class RiskAttribution(PerformanceAttribution):
    """
    风险归因模型
    
    将组合风险分解为各个因子的贡献:
    - 因子风险贡献
    - 特质风险贡献
    - 边际风险贡献
    """

    def __init__(self, factor_cov_matrix: pd.DataFrame, factor_exposures: pd.DataFrame):
        super().__init__()
        self.factor_cov_matrix = factor_cov_matrix
        self.factor_exposures = factor_exposures
    
    def analyze(self, weights: np.ndarray) -> Dict[str, Any]:
        """
        执行风险归因分析
        
        Args:
            weights: 组合权重
            
        Returns:
            风险归因结果
        """
        factor_exposure = self.factor_exposures @ weights
        
        factor_variances = np.diag(self.factor_cov_matrix)
        factor_std_devs = np.sqrt(factor_variances)
        
        factor_risk_contributions = factor_exposure ** 2 * factor_variances
        
        total_factor_risk = np.sqrt(np.sum(factor_risk_contributions))
        
        marginal_risk_contributions = factor_exposure @ self.factor_cov_matrix
        
        risk_contribution_percentages = (factor_risk_contributions / np.sum(factor_risk_contributions)) * 100
        
        return {
            'factor_exposure': factor_exposure,
            'factor_risk_contributions': factor_risk_contributions,
            'total_factor_risk': total_factor_risk,
            'marginal_risk_contributions': marginal_risk_contributions,
            'risk_contribution_percentages': risk_contribution_percentages,
            'risk_attribution_df': pd.DataFrame({
                'exposure': factor_exposure,
                'risk_contribution': factor_risk_contributions,
                'percentage': risk_contribution_percentages
            })
        }


class TimeSeriesAttribution(PerformanceAttribution):
    """
    时序归因模型
    
    在时间维度上分析归因:
    - 滚动归因
    - 事件归因
    - 周期分析
    """

    def __init__(self, window_size: int = 20):
        super().__init__()
        self.window_size = window_size
    
    def analyze(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        factor_returns: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        执行时序归因分析
        
        Args:
            portfolio_returns: 组合收益
            benchmark_returns: 基准收益
            factor_returns: 因子收益（可选）
            
        Returns:
            时序归因结果
        """
        excess_returns = portfolio_returns - benchmark_returns
        
        rolling_excess = excess_returns.rolling(self.window_size).mean()
        rolling_volatility = excess_returns.rolling(self.window_size).std()
        rolling_sharpe = rolling_excess / rolling_volatility
        
        attribution_summary = {
            'excess_returns': excess_returns,
            'rolling_excess': rolling_excess,
            'rolling_volatility': rolling_volatility,
            'rolling_sharpe': rolling_sharpe,
            'period_analysis': self._analyze_periods(excess_returns),
            'event_analysis': self._analyze_events(excess_returns)
        }
        
        return attribution_summary
    
    def _analyze_periods(self, excess_returns: pd.Series) -> Dict[str, Any]:
        """分析不同时间段的表现"""
        monthly = excess_returns.resample('M').sum()
        quarterly = excess_returns.resample('Q').sum()
        yearly = excess_returns.resample('Y').sum()
        
        return {
            'monthly': monthly,
            'quarterly': quarterly,
            'yearly': yearly,
            'best_month': monthly.idxmax(),
            'worst_month': monthly.idxmin(),
            'average_monthly': monthly.mean()
        }
    
    def _analyze_events(self, excess_returns: pd.Series) -> List[Dict[str, Any]]:
        """分析重大事件"""
        significant_events = []
        threshold = excess_returns.std() * 2
        
        for date, return_val in excess_returns.items():
            if abs(return_val) > threshold:
                significant_events.append({
                    'date': date,
                    'excess_return': return_val,
                    'type': 'positive' if return_val > 0 else 'negative',
                    'magnitude': abs(return_val) / excess_returns.std()
                })
        
        return significant_events


class AttributionReport:
    """
    归因报告生成器
    
    生成完整的归因分析报告
    """

    def __init__(self):
        pass
    
    def generate_report(
        self,
        brinson_result: Optional[Dict] = None,
        factor_result: Optional[Dict] = None,
        risk_result: Optional[Dict] = None,
        timeseries_result: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        生成综合归因报告
        
        Args:
            brinson_result: Brinson归因结果
            factor_result: 因子归因结果
            risk_result: 风险归因结果
            timeseries_result: 时序归因结果
            
        Returns:
            综合报告
        """
        report = {
            'generated_at': datetime.now().isoformat(),
            'sections': []
        }
        
        if brinson_result:
            report['sections'].append({
                'name': 'Brinson Attribution',
                'summary': {
                    'total_allocation': brinson_result['total_allocation'],
                    'total_selection': brinson_result['total_selection'],
                    'total_interaction': brinson_result['total_interaction'],
                    'total_excess_return': brinson_result['total_excess_return']
                }
            })
        
        if factor_result:
            report['sections'].append({
                'name': 'Factor Attribution',
                'summary': {
                    'top_factors': factor_result['factor_importance'].head(5).to_dict(),
                    'explained_variance': factor_result['explained_variance']
                }
            })
        
        if risk_result:
            report['sections'].append({
                'name': 'Risk Attribution',
                'summary': {
                    'total_factor_risk': risk_result['total_factor_risk'],
                    'risk_contributions': risk_result['risk_contribution_percentages'].to_dict()
                }
            })
        
        if timeseries_result:
            report['sections'].append({
                'name': 'Time Series Analysis',
                'summary': {
                    'average_rolling_sharpe': timeseries_result['rolling_sharpe'].mean(),
                    'best_month': str(timeseries_result['period_analysis']['best_month']),
                    'worst_month': str(timeseries_result['period_analysis']['worst_month'])
                }
            })
        
        return report


class AttributionFactory:
    """
    归因分析工厂
    
    根据需求创建不同类型的归因分析器
    """

    @staticmethod
    def create_brinson() -> BrinsonAttribution:
        """创建Brinson归因分析器"""
        return BrinsonAttribution()
    
    @staticmethod
    def create_factor(
        factor_returns: pd.DataFrame,
        factor_exposures: pd.DataFrame
    ) -> FactorAttribution:
        """创建因子归因分析器"""
        return FactorAttribution(factor_returns, factor_exposures)
    
    @staticmethod
    def create_risk(
        factor_cov_matrix: pd.DataFrame,
        factor_exposures: pd.DataFrame
    ) -> RiskAttribution:
        """创建风险归因分析器"""
        return RiskAttribution(factor_cov_matrix, factor_exposures)
    
    @staticmethod
    def create_timeseries(window_size: int = 20) -> TimeSeriesAttribution:
        """创建时序归因分析器"""
        return TimeSeriesAttribution(window_size)
    
    @staticmethod
    def create_report() -> AttributionReport:
        """创建归因报告生成器"""
        return AttributionReport()