"""
组合优化层 - Portfolio Optimization

核心功能:
    - 最大化风险调整后收益
    - 支持多种优化目标（Sharpe、IR、收益/风险约束）
    - BARRA风险模型集成
    - 行业/风格暴露控制
    - 约束条件管理

目标函数:
    max_w w^T μ - λ w^T Σ w
    
    即：收益最大，风险最小

这是中国量化私募的核心优化框架。
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds, LinearConstraint
from scipy.linalg import cholesky
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime


class PortfolioOptimizer:
    """
    组合优化器
    
    支持多种优化目标：
    - Sharpe Ratio 最大化
    - Information Ratio 最大化
    - 收益最大化（带风险约束）
    - 风险最小化（带收益约束）
    """

    def __init__(
        self,
        returns: Optional[np.ndarray] = None,
        cov_matrix: Optional[np.ndarray] = None,
        factor_exposures: Optional[np.ndarray] = None,
        factor_cov_matrix: Optional[np.ndarray] = None
    ):
        self.returns = returns
        self.cov_matrix = cov_matrix
        self.factor_exposures = factor_exposures
        self.factor_cov_matrix = factor_cov_matrix
        
        self.n_assets = None
        if returns is not None:
            self.n_assets = len(returns)
        
    def _calculate_portfolio_return(self, weights: np.ndarray) -> float:
        """计算组合收益"""
        return np.dot(weights, self.returns)
    
    def _calculate_portfolio_risk(self, weights: np.ndarray) -> float:
        """计算组合风险（标准差）"""
        if self.factor_exposures is not None and self.factor_cov_matrix is not None:
            return self._calculate_factor_risk(weights)
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
    
    def _calculate_factor_risk(self, weights: np.ndarray) -> float:
        """计算基于因子模型的风险"""
        factor_exposure = np.dot(weights.T, self.factor_exposures)
        return np.sqrt(np.dot(factor_exposure, np.dot(self.factor_cov_matrix, factor_exposure)))
    
    def _calculate_sharpe_ratio(self, weights: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """计算Sharpe Ratio"""
        excess_return = self._calculate_portfolio_return(weights) - risk_free_rate
        risk = self._calculate_portfolio_risk(weights)
        return excess_return / risk if risk > 0 else -np.inf
    
    def _calculate_information_ratio(self, weights: np.ndarray, benchmark_weights: np.ndarray) -> float:
        """计算Information Ratio"""
        active_return = np.dot(weights - benchmark_weights, self.returns)
        active_risk = np.sqrt(np.dot((weights - benchmark_weights).T, 
                                    np.dot(self.cov_matrix, weights - benchmark_weights)))
        return active_return / active_risk if active_risk > 0 else -np.inf
    
    def maximize_sharpe(
        self,
        risk_free_rate: float = 0.0,
        constraints: Optional[List[Dict]] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> np.ndarray:
        """
        最大化Sharpe Ratio
        
        Args:
            risk_free_rate: 无风险收益率
            constraints: 约束条件列表
            bounds: 权重边界
            
        Returns:
            最优权重
        """
        def objective(weights):
            return -self._calculate_sharpe_ratio(weights, risk_free_rate)
        
        return self._optimize(objective, constraints, bounds)
    
    def maximize_return(
        self,
        max_risk: float,
        constraints: Optional[List[Dict]] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> np.ndarray:
        """
        在风险约束下最大化收益
        
        Args:
            max_risk: 最大允许风险
            constraints: 约束条件列表
            bounds: 权重边界
            
        Returns:
            最优权重
        """
        def objective(weights):
            return -self._calculate_portfolio_return(weights)
        
        risk_constraint = {
            'type': 'ineq',
            'fun': lambda w: max_risk - self._calculate_portfolio_risk(w)
        }
        
        all_constraints = [risk_constraint]
        if constraints:
            all_constraints.extend(constraints)
        
        return self._optimize(objective, all_constraints, bounds)
    
    def minimize_risk(
        self,
        min_return: float,
        constraints: Optional[List[Dict]] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> np.ndarray:
        """
        在收益约束下最小化风险
        
        Args:
            min_return: 最小目标收益
            constraints: 约束条件列表
            bounds: 权重边界
            
        Returns:
            最优权重
        """
        def objective(weights):
            return self._calculate_portfolio_risk(weights)
        
        return_constraint = {
            'type': 'ineq',
            'fun': lambda w: self._calculate_portfolio_return(w) - min_return
        }
        
        all_constraints = [return_constraint]
        if constraints:
            all_constraints.extend(constraints)
        
        return self._optimize(objective, all_constraints, bounds)
    
    def maximize_information_ratio(
        self,
        benchmark_weights: np.ndarray,
        constraints: Optional[List[Dict]] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> np.ndarray:
        """
        最大化Information Ratio
        
        Args:
            benchmark_weights: 基准权重
            constraints: 约束条件列表
            bounds: 权重边界
            
        Returns:
            最优权重
        """
        def objective(weights):
            return -self._calculate_information_ratio(weights, benchmark_weights)
        
        return self._optimize(objective, constraints, bounds)
    
    def _optimize(
        self,
        objective,
        constraints: Optional[List[Dict]] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> np.ndarray:
        """执行优化"""
        if self.n_assets is None:
            raise ValueError("No returns data provided")
        
        initial_weights = np.ones(self.n_assets) / self.n_assets
        
        default_bounds = Bounds(np.zeros(self.n_assets), np.ones(self.n_assets))
        
        if bounds is not None:
            custom_bounds = Bounds(bounds[0], bounds[1])
        else:
            custom_bounds = default_bounds
        
        weight_sum_constraint = {
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0
        }
        
        all_constraints = [weight_sum_constraint]
        if constraints:
            all_constraints.extend(constraints)
        
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=custom_bounds,
            constraints=all_constraints,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        if result.success:
            return result.x
        else:
            print(f"Optimization failed: {result.message}")
            return initial_weights


class BarraRiskModel:
    """
    BARRA风格风险模型
    
    支持的风格因子:
    - Size: 市值
    - Value: 价值
    - Momentum: 动量
    - Volatility: 波动率
    - Quality: 质量
    - Growth: 成长
    - Liquidity: 流动性
    - Leverage: 杠杆
    - Dividend: 股息
    """

    def __init__(self):
        self.factor_names = [
            'Size', 'Value', 'Momentum', 'Volatility',
            'Quality', 'Growth', 'Liquidity', 'Leverage', 'Dividend'
        ]
        
        self.factor_exposures = None
        self.factor_cov_matrix = None
        self.idiosyncratic_variances = None
        
    def fit(self, returns: pd.DataFrame, factor_data: pd.DataFrame):
        """
        拟合风险模型
        
        Args:
            returns: 股票收益率 [n_dates, n_stocks]
            factor_data: 因子暴露数据 [n_stocks, n_factors]
        """
        self.factor_exposures = factor_data.values
        
        factor_returns = self._calculate_factor_returns(returns, factor_data)
        
        self.factor_cov_matrix = np.cov(factor_returns.T)
        
        self.idiosyncratic_variances = self._calculate_idiosyncratic_variances(returns, factor_data)
    
    def _calculate_factor_returns(self, returns: pd.DataFrame, factor_data: pd.DataFrame) -> np.ndarray:
        """计算因子收益率"""
        exposures = factor_data.values
        pinv_exposures = np.linalg.pinv(exposures.T @ exposures) @ exposures.T
        return pinv_exposures @ returns.values.T
    
    def _calculate_idiosyncratic_variances(self, returns: pd.DataFrame, factor_data: pd.DataFrame) -> np.ndarray:
        """计算特质方差"""
        exposures = factor_data.values
        factor_returns = self._calculate_factor_returns(returns, factor_data)
        fitted_returns = exposures @ factor_returns
        residuals = returns.values.T - fitted_returns
        return np.var(residuals, axis=1)
    
    def calculate_risk(self, weights: np.ndarray) -> float:
        """计算组合风险"""
        factor_exposure = np.dot(weights.T, self.factor_exposures)
        factor_risk = np.sqrt(np.dot(factor_exposure, np.dot(self.factor_cov_matrix, factor_exposure)))
        
        specific_risk = np.sqrt(np.dot(weights.T, weights * self.idiosyncratic_variances))
        
        total_risk = np.sqrt(factor_risk ** 2 + specific_risk ** 2)
        
        return total_risk
    
    def get_factor_exposures(self, weights: np.ndarray) -> np.ndarray:
        """获取组合因子暴露"""
        return np.dot(weights.T, self.factor_exposures)


class ConstrainedOptimizer:
    """
    带约束的组合优化器
    
    支持的约束类型:
    - 行业暴露约束
    - 风格因子暴露约束
    - 单个股票权重约束
    - 最大持仓数量约束
    - 中性化约束
    """

    def __init__(self):
        self.constraints = []
        
    def add_sector_constraint(
        self,
        sector_mask: np.ndarray,
        min_weight: float = 0.0,
        max_weight: float = 1.0
    ):
        """
        添加行业约束
        
        Args:
            sector_mask: 行业成员mask [n_stocks], True表示属于该行业
            min_weight: 最小行业权重
            max_weight: 最大行业权重
        """
        self.constraints.append({
            'type': 'ineq',
            'fun': lambda w: np.sum(w * sector_mask) - min_weight
        })
        self.constraints.append({
            'type': 'ineq',
            'fun': lambda w: max_weight - np.sum(w * sector_mask)
        })
    
    def add_factor_constraint(
        self,
        factor_exposures: np.ndarray,
        min_exposure: float = -0.1,
        max_exposure: float = 0.1
    ):
        """
        添加因子暴露约束
        
        Args:
            factor_exposures: 因子暴露 [n_stocks]
            min_exposure: 最小暴露
            max_exposure: 最大暴露
        """
        self.constraints.append({
            'type': 'ineq',
            'fun': lambda w: np.dot(w, factor_exposures) - min_exposure
        })
        self.constraints.append({
            'type': 'ineq',
            'fun': lambda w: max_exposure - np.dot(w, factor_exposures)
        })
    
    def add_long_short_constraint(self, long_mask: np.ndarray, short_mask: np.ndarray):
        """
        添加多空约束
        
        Args:
            long_mask: 多头股票mask
            short_mask: 空头股票mask
        """
        self.constraints.append({
            'type': 'ineq',
            'fun': lambda w: w[long_mask]  # 多头必须为正
        })
        self.constraints.append({
            'type': 'ineq',
            'fun': lambda w: -w[short_mask]  # 空头必须为负
        })
    
    def add_neutral_constraint(self, factor_exposures: np.ndarray):
        """
        添加中性化约束
        
        Args:
            factor_exposures: 因子暴露 [n_stocks]
        """
        self.constraints.append({
            'type': 'eq',
            'fun': lambda w: np.dot(w, factor_exposures)
        })
    
    def add_position_limit(self, max_positions: int, n_stocks: int):
        """
        添加最大持仓数量约束
        
        Args:
            max_positions: 最大持仓数量
            n_stocks: 股票总数
        """
        self.constraints.append({
            'type': 'ineq',
            'fun': lambda w: max_positions - np.sum(w > 0.001)
        })
    
    def get_constraints(self) -> List[Dict]:
        """获取所有约束"""
        return self.constraints


class MeanVarianceOptimizer(PortfolioOptimizer):
    """
    均值-方差优化器
    
    经典的Markowitz优化框架
    """

    def __init__(
        self,
        returns: np.ndarray,
        cov_matrix: np.ndarray,
        lambda_reg: float = 1.0
    ):
        super().__init__(returns, cov_matrix)
        self.lambda_reg = lambda_reg
    
    def optimize(
        self,
        constraints: Optional[List[Dict]] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> np.ndarray:
        """
        执行均值-方差优化
        
        目标函数: max w^T μ - λ w^T Σ w
        
        Args:
            constraints: 约束条件
            bounds: 权重边界
            
        Returns:
            最优权重
        """
        def objective(weights):
            return -(np.dot(weights, self.returns) - 
                     self.lambda_reg * np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        
        return self._optimize(objective, constraints, bounds)
    
    def efficient_frontier(
        self,
        n_points: int = 20,
        constraints: Optional[List[Dict]] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算有效前沿
        
        Args:
            n_points: 前沿上的点数
            constraints: 约束条件
            bounds: 权重边界
            
        Returns:
            风险数组, 收益数组, 权重数组
        """
        risks = []
        returns = []
        weights_list = []
        
        lambda_values = np.logspace(-2, 2, n_points)
        
        for lambda_val in lambda_values:
            self.lambda_reg = lambda_val
            weights = self.optimize(constraints, bounds)
            risk = self._calculate_portfolio_risk(weights)
            ret = self._calculate_portfolio_return(weights)
            
            risks.append(risk)
            returns.append(ret)
            weights_list.append(weights)
        
        return np.array(risks), np.array(returns), np.array(weights_list)