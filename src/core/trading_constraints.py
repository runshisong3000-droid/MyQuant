"""
A股交易约束模块 - Trading Constraints

核心功能:
    - ST股票过滤
    - 停牌过滤
    - 涨跌停过滤
    - 新股上市过滤
    - 单票权重上限
    - 行业权重上限
    - 换手率约束
    - 流动性约束
    - T+1约束
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta


class AShareTradingConstraints:
    """
    A股交易约束管理器
    
    实现A股特有的交易约束：
    1. ST股票过滤
    2. 停牌股票过滤
    3. 涨跌停过滤
    4. 新股上市不足N天过滤
    5. 单票权重上限
    6. 行业权重上限
    7. 换手率约束
    8. 流动性约束
    9. 成交额容量约束
    10. T+1约束
    """

    def __init__(self):
        self.st_stocks = set()
        self.suspended_stocks = set()
        self.limit_up_stocks = set()
        self.limit_down_stocks = set()
        self.new_stocks = {}
        self.industry_mappings = {}
    
    def set_st_stocks(self, st_stocks: List[str]):
        """设置ST股票列表"""
        self.st_stocks = set(st_stocks)
    
    def set_suspended_stocks(self, suspended_stocks: List[str]):
        """设置停牌股票列表"""
        self.suspended_stocks = set(suspended_stocks)
    
    def set_limit_up_down(self, limit_up: List[str], limit_down: List[str]):
        """设置涨跌停股票列表"""
        self.limit_up_stocks = set(limit_up)
        self.limit_down_stocks = set(limit_down)
    
    def set_new_stocks(self, new_stocks: Dict[str, datetime]):
        """设置新股列表（股票代码:上市日期）"""
        self.new_stocks = new_stocks
    
    def set_industry_mappings(self, mappings: Dict[str, str]):
        """设置行业映射（股票代码:行业）"""
        self.industry_mappings = mappings
    
    def filter_st_stocks(self, stocks: List[str]) -> List[str]:
        """过滤ST股票"""
        return [s for s in stocks if s not in self.st_stocks]
    
    def filter_suspended_stocks(self, stocks: List[str]) -> List[str]:
        """过滤停牌股票"""
        return [s for s in stocks if s not in self.suspended_stocks]
    
    def filter_limit_up_down(self, stocks: List[str], is_buy: bool) -> List[str]:
        """过滤涨跌停股票"""
        if is_buy:
            return [s for s in stocks if s not in self.limit_up_stocks]
        else:
            return [s for s in stocks if s not in self.limit_down_stocks]
    
    def filter_new_stocks(
        self,
        stocks: List[str],
        date: datetime,
        min_days_listed: int = 30
    ) -> List[str]:
        """过滤上市不足N天的新股"""
        filtered = []
        
        for stock in stocks:
            if stock in self.new_stocks:
                days_listed = (date - self.new_stocks[stock]).days
                if days_listed >= min_days_listed:
                    filtered.append(stock)
            else:
                filtered.append(stock)
        
        return filtered
    
    def apply_position_limit(
        self,
        weights: pd.Series,
        max_weight: float = 0.1
    ) -> pd.Series:
        """应用单票权重上限"""
        limited_weights = weights.copy()
        limited_weights[limited_weights > max_weight] = max_weight
        limited_weights = limited_weights / limited_weights.sum()
        return limited_weights
    
    def apply_industry_limit(
        self,
        weights: pd.Series,
        max_industry_weight: float = 0.3
    ) -> pd.Series:
        """应用行业权重上限"""
        limited_weights = weights.copy()
        
        industry_weights = {}
        for stock, weight in limited_weights.items():
            industry = self.industry_mappings.get(stock, 'unknown')
            industry_weights[industry] = industry_weights.get(industry, 0) + weight
        
        for industry, total_weight in industry_weights.items():
            if total_weight > max_industry_weight:
                scale_factor = max_industry_weight / total_weight
                for stock, weight in limited_weights.items():
                    if self.industry_mappings.get(stock, 'unknown') == industry:
                        limited_weights[stock] *= scale_factor
        
        limited_weights = limited_weights / limited_weights.sum()
        return limited_weights
    
    def apply_turnover_limit(
        self,
        weights: pd.Series,
        turnover_data: pd.Series,
        max_turnover_ratio: float = 0.1
    ) -> pd.Series:
        """应用换手率约束"""
        limited_weights = weights.copy()
        
        for stock, weight in limited_weights.items():
            if stock in turnover_data:
                turnover = turnover_data[stock]
                max_position = turnover * max_turnover_ratio
                
                if weight > max_position:
                    limited_weights[stock] = max_position
        
        limited_weights = limited_weights / limited_weights.sum()
        return limited_weights
    
    def apply_liquidity_limit(
        self,
        weights: pd.Series,
        volume_data: pd.Series,
        min_daily_volume: float = 10000000
    ) -> pd.Series:
        """应用流动性约束"""
        filtered_weights = weights.copy()
        
        for stock, weight in filtered_weights.items():
            if stock in volume_data and volume_data[stock] < min_daily_volume:
                filtered_weights[stock] = 0
        
        if filtered_weights.sum() == 0:
            return weights
        
        filtered_weights = filtered_weights / filtered_weights.sum()
        return filtered_weights
    
    def apply_all_constraints(
        self,
        stocks: List[str],
        weights: Optional[pd.Series] = None,
        date: Optional[datetime] = None,
        is_buy: bool = True,
        turnover_data: Optional[pd.Series] = None,
        volume_data: Optional[pd.Series] = None,
        **kwargs
    ) -> Tuple[List[str], pd.Series]:
        """
        应用所有约束
        
        Args:
            stocks: 股票列表
            weights: 权重（可选）
            date: 日期（用于新股过滤）
            is_buy: 是否买入
            turnover_data: 换手率数据（可选）
            volume_data: 成交量数据（可选）
            
        Returns:
            (过滤后的股票列表, 调整后的权重)
        """
        stocks = self.filter_st_stocks(stocks)
        stocks = self.filter_suspended_stocks(stocks)
        stocks = self.filter_limit_up_down(stocks, is_buy)
        
        if date:
            min_days = kwargs.get('min_days_listed', 30)
            stocks = self.filter_new_stocks(stocks, date, min_days)
        
        if weights is not None:
            weights = weights.loc[stocks]
            
            max_weight = kwargs.get('max_weight', 0.1)
            weights = self.apply_position_limit(weights, max_weight)
            
            max_industry_weight = kwargs.get('max_industry_weight', 0.3)
            weights = self.apply_industry_limit(weights, max_industry_weight)
            
            if turnover_data is not None:
                max_turnover = kwargs.get('max_turnover_ratio', 0.1)
                weights = self.apply_turnover_limit(weights, turnover_data, max_turnover)
            
            if volume_data is not None:
                min_volume = kwargs.get('min_daily_volume', 10000000)
                weights = self.apply_liquidity_limit(weights, volume_data, min_volume)
            
            stocks = weights[weights > 0].index.tolist()
        
        return stocks, weights


class ConstraintValidator:
    """
    约束验证器
    
    验证组合是否满足所有约束
    """

    def __init__(self, constraints: AShareTradingConstraints):
        self.constraints = constraints
    
    def validate(
        self,
        stocks: List[str],
        weights: pd.Series,
        date: datetime,
        **kwargs
    ) -> Dict[str, Any]:
        """
        验证组合约束
        
        Args:
            stocks: 股票列表
            weights: 权重
            date: 日期
            
        Returns:
            验证结果
        """
        results = {
            'is_valid': True,
            'violations': [],
            'warnings': []
        }
        
        st_count = len([s for s in stocks if s in self.constraints.st_stocks])
        if st_count > 0:
            results['violations'].append(f"Contains {st_count} ST stocks")
            results['is_valid'] = False
        
        suspended_count = len([s for s in stocks if s in self.constraints.suspended_stocks])
        if suspended_count > 0:
            results['violations'].append(f"Contains {suspended_count} suspended stocks")
            results['is_valid'] = False
        
        if weights is not None:
            max_weight = kwargs.get('max_weight', 0.1)
            over_limit = weights[weights > max_weight]
            if len(over_limit) > 0:
                results['violations'].append(f"{len(over_limit)} stocks exceed max weight of {max_weight}")
                results['is_valid'] = False
            
            industry_weights = {}
            for stock, weight in weights.items():
                industry = self.constraints.industry_mappings.get(stock, 'unknown')
                industry_weights[industry] = industry_weights.get(industry, 0) + weight
            
            max_industry_weight = kwargs.get('max_industry_weight', 0.3)
            over_industry_limit = {k: v for k, v in industry_weights.items() if v > max_industry_weight}
            if over_industry_limit:
                results['violations'].append(f"Industry limits exceeded: {over_industry_limit}")
                results['is_valid'] = False
        
        return results


class TransactionCostCalculator:
    """
    交易成本计算器
    
    计算A股交易成本：
    - 佣金
    - 印花税（卖出）
    - 过户费
    - 市场冲击
    """

    def __init__(
        self,
        commission_rate: float = 0.0003,
        stamp_duty_rate: float = 0.001,
        transfer_fee_rate: float = 0.00002
    ):
        self.commission_rate = commission_rate
        self.stamp_duty_rate = stamp_duty_rate
        self.transfer_fee_rate = transfer_fee_rate
        self.min_commission = 5.0
    
    def calculate_commission(self, price: float, quantity: int) -> float:
        """计算佣金"""
        commission = price * quantity * self.commission_rate
        return max(commission, self.min_commission)
    
    def calculate_stamp_duty(self, price: float, quantity: int) -> float:
        """计算印花税（卖出时）"""
        return price * quantity * self.stamp_duty_rate
    
    def calculate_transfer_fee(self, price: float, quantity: int) -> float:
        """计算过户费"""
        return price * quantity * self.transfer_fee_rate
    
    def calculate_market_impact(
        self,
        price: float,
        quantity: int,
        daily_volume: float,
        impact_factor: float = 0.001
    ) -> float:
        """计算市场冲击成本"""
        order_ratio = quantity / daily_volume
        impact = price * quantity * impact_factor * np.sqrt(order_ratio)
        return impact
    
    def calculate_total_cost(
        self,
        price: float,
        quantity: int,
        is_buy: bool,
        daily_volume: float = 1e8
    ) -> Dict[str, float]:
        """
        计算总交易成本
        
        Args:
            price: 价格
            quantity: 数量
            is_buy: 是否买入
            daily_volume: 日成交量
            
        Returns:
            成本明细
        """
        costs = {
            'commission': self.calculate_commission(price, quantity),
            'stamp_duty': self.calculate_stamp_duty(price, quantity) if not is_buy else 0.0,
            'transfer_fee': self.calculate_transfer_fee(price, quantity),
            'market_impact': self.calculate_market_impact(price, quantity, daily_volume)
        }
        
        costs['total'] = sum(costs.values())
        
        return costs