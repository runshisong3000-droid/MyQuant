"""
风控引擎 - Risk Engine

核心功能:
    - BARRA风险模型
    - 风格暴露控制
    - 行业暴露控制
    - 最大回撤控制
    - 中性化处理
    - 实时风险监控

这是量化私募最重视的环节：风控先于收益

核心目标:
    不是 maximize(return)
    而是 maximize(Sharpe) 或 maximize(IR)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import warnings


class RiskEngine:
    """
    风控引擎
    
    核心职责:
    1. 计算风险指标
    2. 监控风险暴露
    3. 执行风险约束
    4. 生成风险报告
    """

    def __init__(self):
        self.risk_model = None
        self.constraints = {}
        self.risk_limits = {}
        
    def set_risk_model(self, risk_model):
        """设置风险模型"""
        self.risk_model = risk_model
        
    def add_risk_limit(self, risk_type: str, limit: float):
        """添加风险限制"""
        self.risk_limits[risk_type] = limit
        
    def calculate_portfolio_risk(self, weights: np.ndarray) -> Dict[str, float]:
        """计算组合风险指标"""
        if self.risk_model is None:
            raise ValueError("Risk model not set")
        
        results = {}
        
        total_risk = self.risk_model.calculate_risk(weights)
        results['total_risk'] = total_risk
        
        factor_exposures = self.risk_model.get_factor_exposures(weights)
        results['factor_exposures'] = dict(zip(self.risk_model.factor_names, factor_exposures))
        
        for i, factor_name in enumerate(self.risk_model.factor_names):
            factor_risk = self._calculate_factor_contribution(weights, i)
            results[f'{factor_name}_risk'] = factor_risk
        
        return results
    
    def _calculate_factor_contribution(self, weights: np.ndarray, factor_idx: int) -> float:
        """计算单个因子的风险贡献"""
        factor_exposure = np.dot(weights, self.risk_model.factor_exposures[:, factor_idx])
        factor_volatility = np.sqrt(self.risk_model.factor_cov_matrix[factor_idx, factor_idx])
        return abs(factor_exposure * factor_volatility)
    
    def check_constraints(self, weights: np.ndarray) -> Tuple[bool, List[str]]:
        """检查风险约束"""
        violations = []
        is_valid = True
        
        risk_metrics = self.calculate_portfolio_risk(weights)
        
        if 'total_risk' in self.risk_limits:
            if risk_metrics['total_risk'] > self.risk_limits['total_risk']:
                violations.append(f"Total risk exceeds limit: {risk_metrics['total_risk']:.4f} > {self.risk_limits['total_risk']:.4f}")
                is_valid = False
        
        for factor_name in self.risk_model.factor_names:
            limit_key = f'{factor_name}_limit'
            if limit_key in self.risk_limits:
                exposure = risk_metrics['factor_exposures'][factor_name]
                limit = self.risk_limits[limit_key]
                if abs(exposure) > limit:
                    violations.append(f"{factor_name} exposure exceeds limit: {exposure:.4f} > {limit:.4f}")
                    is_valid = False
        
        return is_valid, violations
    
    def enforce_constraints(self, weights: np.ndarray) -> np.ndarray:
        """强制执行风险约束"""
        adjusted_weights = weights.copy()
        
        risk_metrics = self.calculate_portfolio_risk(adjusted_weights)
        
        for factor_name in self.risk_model.factor_names:
            limit_key = f'{factor_name}_limit'
            if limit_key in self.risk_limits:
                exposure = risk_metrics['factor_exposures'][factor_name]
                limit = self.risk_limits[limit_key]
                
                if abs(exposure) > limit:
                    adjustment = (limit / abs(exposure)) if exposure != 0 else 0
                    adjusted_weights *= adjustment
        
        adjusted_weights = adjusted_weights / np.sum(adjusted_weights)
        
        return adjusted_weights
    
    def generate_risk_report(self, weights: np.ndarray) -> Dict[str, Any]:
        """生成风险报告"""
        report = {}
        
        report['timestamp'] = datetime.now().isoformat()
        
        risk_metrics = self.calculate_portfolio_risk(weights)
        report['risk_metrics'] = risk_metrics
        
        is_valid, violations = self.check_constraints(weights)
        report['constraints_check'] = {
            'is_valid': is_valid,
            'violations': violations
        }
        
        report['weights_summary'] = {
            'n_positions': np.sum(weights > 0.001),
            'max_weight': np.max(weights),
            'min_weight': np.min(weights),
            'avg_weight': np.mean(weights)
        }
        
        return report


class ExposureMonitor:
    """
    暴露监控器
    
    监控组合在各种维度上的暴露:
    - 风格因子暴露
    - 行业暴露
    - 市值分布
    - 流动性风险
    """

    def __init__(self):
        self.sector_mappings = {}
        self.market_cap_data = {}
        
    def set_sector_mappings(self, mappings: Dict[str, List[str]]):
        """设置行业映射"""
        self.sector_mappings = mappings
        
    def set_market_cap_data(self, market_cap: Dict[str, float]):
        """设置市值数据"""
        self.market_cap_data = market_cap
        
    def calculate_sector_exposure(self, weights: pd.Series) -> pd.Series:
        """计算行业暴露"""
        sector_exposures = pd.Series(index=self.sector_mappings.keys(), dtype=float)
        
        for sector, stocks in self.sector_mappings.items():
            sector_exposures[sector] = weights.loc[weights.index.isin(stocks)].sum()
        
        return sector_exposures.fillna(0)
    
    def calculate_market_cap_distribution(self, weights: pd.Series) -> Dict[str, float]:
        """计算市值分布"""
        cap_groups = {
            'large': 0.0,
            'mid': 0.0,
            'small': 0.0
        }
        
        total_cap = sum(self.market_cap_data.values())
        
        for stock, weight in weights.items():
            if stock in self.market_cap_data:
                cap = self.market_cap_data[stock]
                cap_ratio = cap / total_cap
                
                if cap_ratio > 0.01:
                    cap_groups['large'] += weight
                elif cap_ratio > 0.001:
                    cap_groups['mid'] += weight
                else:
                    cap_groups['small'] += weight
        
        return cap_groups
    
    def monitor(self, weights: pd.Series) -> Dict[str, Any]:
        """执行监控"""
        report = {}
        
        report['sector_exposures'] = self.calculate_sector_exposure(weights).to_dict()
        report['market_cap_distribution'] = self.calculate_market_cap_distribution(weights)
        
        report['concentration_risk'] = {
            'top_10_weight': weights.nlargest(10).sum(),
            'herfindahl_index': (weights ** 2).sum()
        }
        
        return report


class DrawdownControl:
    """
    最大回撤控制器
    
    监控和控制组合的最大回撤
    """

    def __init__(self, max_drawdown: float = 0.1):
        self.max_drawdown = max_drawdown
        self.peak_value = 1.0
        self.current_value = 1.0
        self.historical_drawdowns = []
        
    def update(self, portfolio_value: float):
        """更新组合价值"""
        self.current_value = portfolio_value
        
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        
        drawdown = (self.peak_value - portfolio_value) / self.peak_value
        self.historical_drawdowns.append(drawdown)
        
    def check_drawdown(self) -> Tuple[bool, float]:
        """检查是否超过最大回撤"""
        current_drawdown = (self.peak_value - self.current_value) / self.peak_value
        is_exceeded = current_drawdown > self.max_drawdown
        
        return is_exceeded, current_drawdown
    
    def get_reduction_factor(self) -> float:
        """计算仓位缩减因子"""
        current_drawdown = (self.peak_value - self.current_value) / self.peak_value
        
        if current_drawdown < 0.5 * self.max_drawdown:
            return 1.0
        elif current_drawdown < self.max_drawdown:
            return 1.0 - (current_drawdown / self.max_drawdown)
        else:
            return 0.5
    
    def reset(self):
        """重置状态"""
        self.peak_value = 1.0
        self.current_value = 1.0
        self.historical_drawdowns = []


class NeutralizationEngine:
    """
    中性化引擎
    
    将组合在指定因子上进行中性化处理:
    - 行业中性化
    - 风格中性化
    - 市场中性化
    """

    def __init__(self):
        pass
    
    def neutralize(
        self,
        weights: np.ndarray,
        factor_exposures: np.ndarray,
        target_exposure: float = 0.0
    ) -> np.ndarray:
        """
        中性化处理
        
        Args:
            weights: 当前权重
            factor_exposures: 因子暴露矩阵 [n_stocks, n_factors]
            target_exposure: 目标暴露值
            
        Returns:
            中性化后的权重
        """
        n_stocks = len(weights)
        n_factors = factor_exposures.shape[1]
        
        A = np.column_stack([np.ones(n_stocks), factor_exposures])
        b = np.array([1.0] + [target_exposure] * n_factors)
        
        AtA_inv = np.linalg.inv(A.T @ A)
        optimal_weights = AtA_inv @ A.T @ weights
        
        return optimal_weights
    
    def industry_neutralize(
        self,
        weights: pd.Series,
        sector_labels: pd.Series
    ) -> pd.Series:
        """
        行业中性化
        
        Args:
            weights: 当前权重
            sector_labels: 行业标签
            
        Returns:
            行业中性化后的权重
        """
        neutralized_weights = weights.copy()
        
        for sector in sector_labels.unique():
            sector_mask = sector_labels == sector
            sector_weights = weights[sector_mask]
            
            if sector_weights.sum() != 0:
                neutralized_weights[sector_mask] = sector_weights / sector_weights.sum()
        
        total_weight = neutralized_weights.sum()
        if total_weight > 0:
            neutralized_weights = neutralized_weights / total_weight
        
        return neutralized_weights
    
    def market_neutralize(
        self,
        long_weights: pd.Series,
        short_weights: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """
        市场中性化
        
        Args:
            long_weights: 多头权重
            short_weights: 空头权重
            
        Returns:
            市场中性化后的多空头权重
        """
        long_total = long_weights.sum()
        short_total = short_weights.sum()
        
        scaling_factor = min(long_total, short_total)
        
        if scaling_factor > 0:
            long_weights = long_weights * (scaling_factor / long_total)
            short_weights = short_weights * (scaling_factor / short_total)
        
        return long_weights, short_weights


class VaRCalculator:
    """
    VaR计算器
    
    计算在险价值:
    - 参数法VaR
    - 历史法VaR
    - Monte Carlo VaR
    """

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        
    def parametric_var(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
        cov_matrix: np.ndarray
    ) -> float:
        """
        参数法VaR
        
        Args:
            weights: 组合权重
            returns: 预期收益率
            cov_matrix: 协方差矩阵
            
        Returns:
            VaR值
        """
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        from scipy.stats import norm
        z_score = norm.ppf(self.confidence_level)
        
        return z_score * portfolio_std
    
    def historical_var(self, portfolio_returns: np.ndarray) -> float:
        """
        历史法VaR
        
        Args:
            portfolio_returns: 历史组合收益率
            
        Returns:
            VaR值
        """
        sorted_returns = np.sort(portfolio_returns)
        var_index = int((1 - self.confidence_level) * len(sorted_returns))
        
        return -sorted_returns[var_index]
    
    def monte_carlo_var(
        self,
        weights: np.ndarray,
        mean_returns: np.ndarray,
        cov_matrix: np.ndarray,
        n_simulations: int = 10000
    ) -> float:
        """
        Monte Carlo VaR
        
        Args:
            weights: 组合权重
            mean_returns: 预期收益率
            cov_matrix: 协方差矩阵
            n_simulations: 模拟次数
            
        Returns:
            VaR值
        """
        n_assets = len(weights)
        
        np.random.seed(42)
        random_returns = np.random.multivariate_normal(
            mean_returns, cov_matrix, n_simulations
        )
        
        portfolio_returns = np.dot(random_returns, weights)
        
        sorted_returns = np.sort(portfolio_returns)
        var_index = int((1 - self.confidence_level) * n_simulations)
        
        return -sorted_returns[var_index]