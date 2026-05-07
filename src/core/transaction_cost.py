"""
交易成本模型 - Transaction Cost Model

核心功能:
    - 佣金计算
    - 滑点模型
    - 市场冲击模型
    - 流动性成本
    - 卖空成本
    - 综合交易成本估算

这是回测引擎的关键组成部分，直接影响策略收益的真实性。
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Any
from datetime import datetime


class TransactionCostModel:
    """
    交易成本模型基类
    
    交易成本组成:
    1. 佣金 (Commission) - 券商收取的费用
    2. 印花税 (Stamp Duty) - 卖出时收取
    3. 滑点 (Slippage) - 实际成交价格与预期价格的差异
    4. 市场冲击 (Market Impact) - 大额订单对市场价格的影响
    5. 流动性成本 (Liquidity Cost) - 买卖价差成本
    """

    def __init__(self):
        self.commission_rate = 0.0003
        self.stamp_duty_rate = 0.001
        self.min_commission = 5.0
        
    def calculate_cost(
        self,
        price: float,
        quantity: int,
        is_buy: bool,
        market_data: Optional[Dict] = None
    ) -> float:
        """
        计算交易成本
        
        Args:
            price: 交易价格
            quantity: 交易数量
            is_buy: 是否买入
            market_data: 市场数据（包含成交量、买卖盘等）
            
        Returns:
            总成本
        """
        total_cost = 0.0
        
        total_cost += self.calculate_commission(price, quantity)
        
        if not is_buy:
            total_cost += self.calculate_stamp_duty(price, quantity)
        
        total_cost += self.calculate_slippage(price, quantity, is_buy, market_data)
        
        total_cost += self.calculate_market_impact(price, quantity, is_buy, market_data)
        
        return total_cost
    
    def calculate_commission(self, price: float, quantity: int) -> float:
        """计算佣金"""
        commission = price * quantity * self.commission_rate
        return max(commission, self.min_commission)
    
    def calculate_stamp_duty(self, price: float, quantity: int) -> float:
        """计算印花税（卖出时）"""
        return price * quantity * self.stamp_duty_rate
    
    def calculate_slippage(self, price: float, quantity: int, is_buy: bool, market_data: Optional[Dict]) -> float:
        """计算滑点"""
        return 0.0
    
    def calculate_market_impact(self, price: float, quantity: int, is_buy: bool, market_data: Optional[Dict]) -> float:
        """计算市场冲击"""
        return 0.0


class FixedSlippageModel(TransactionCostModel):
    """
    固定滑点模型
    
    假设滑点为固定点数或固定比例
    """

    def __init__(self, slippage_bps: float = 1.0):
        super().__init__()
        self.slippage_bps = slippage_bps / 10000  # 转换为比例
    
    def calculate_slippage(self, price: float, quantity: int, is_buy: bool, market_data: Optional[Dict]) -> float:
        """计算固定滑点"""
        slippage = price * quantity * self.slippage_bps
        return slippage


class VolumeWeightedSlippageModel(TransactionCostModel):
    """
    成交量加权滑点模型
    
    滑点与订单大小占市场成交量的比例相关
    """

    def __init__(self, alpha: float = 0.1, beta: float = 0.01):
        super().__init__()
        self.alpha = alpha  # 冲击系数
        self.beta = beta    # 最小滑点
    
    def calculate_slippage(self, price: float, quantity: int, is_buy: bool, market_data: Optional[Dict]) -> float:
        """计算成交量加权滑点"""
        if market_data is None or 'volume' not in market_data:
            return price * quantity * self.beta
        
        market_volume = market_data.get('volume', 1)
        order_ratio = min(quantity / market_volume, 0.1)
        
        slippage = price * quantity * (self.beta + self.alpha * order_ratio)
        return slippage


class MarketImpactModel(TransactionCostModel):
    """
    市场冲击模型
    
    基于 Almgren-Chriss 模型的简化版本
    """

    def __init__(self, permanent_impact: float = 0.001, temporary_impact: float = 0.002):
        super().__init__()
        self.permanent_impact = permanent_impact
        self.temporary_impact = temporary_impact
    
    def calculate_market_impact(self, price: float, quantity: int, is_buy: bool, market_data: Optional[Dict]) -> float:
        """计算市场冲击"""
        if market_data is None or 'volume' not in market_data:
            return 0.0
        
        market_volume = market_data.get('volume', 1)
        daily_volume = market_data.get('daily_volume', market_volume * 240)
        
        order_ratio = quantity / daily_volume
        
        impact = self.permanent_impact * np.sqrt(order_ratio) + self.temporary_impact * order_ratio
        
        return price * quantity * impact


class SpreadCostModel(TransactionCostModel):
    """
    买卖价差成本模型
    
    考虑买卖盘之间的价差
    """

    def __init__(self):
        super().__init__()
    
    def calculate_slippage(self, price: float, quantity: int, is_buy: bool, market_data: Optional[Dict]) -> float:
        """计算价差成本"""
        if market_data is None:
            return 0.0
        
        bid = market_data.get('bid', price)
        ask = market_data.get('ask', price)
        
        if is_buy:
            effective_price = ask
        else:
            effective_price = bid
        
        price_diff = abs(effective_price - price)
        return price_diff * quantity


class ComprehensiveCostModel(TransactionCostModel):
    """
    综合交易成本模型
    
    整合所有成本因素:
    - 佣金
    - 印花税
    - 滑点（成交量加权）
    - 市场冲击
    - 流动性成本
    """

    def __init__(
        self,
        commission_rate: float = 0.0003,
        stamp_duty_rate: float = 0.001,
        min_commission: float = 5.0,
        slippage_alpha: float = 0.1,
        slippage_beta: float = 0.01,
        permanent_impact: float = 0.001,
        temporary_impact: float = 0.002
    ):
        super().__init__()
        self.commission_rate = commission_rate
        self.stamp_duty_rate = stamp_duty_rate
        self.min_commission = min_commission
        self.slippage_alpha = slippage_alpha
        self.slippage_beta = slippage_beta
        self.permanent_impact = permanent_impact
        self.temporary_impact = temporary_impact
    
    def calculate_cost(
        self,
        price: float,
        quantity: int,
        is_buy: bool,
        market_data: Optional[Dict] = None
    ) -> float:
        """计算综合交易成本"""
        total_cost = 0.0
        
        total_cost += self.calculate_commission(price, quantity)
        
        if not is_buy:
            total_cost += self.calculate_stamp_duty(price, quantity)
        
        total_cost += self.calculate_slippage(price, quantity, is_buy, market_data)
        
        total_cost += self.calculate_market_impact(price, quantity, is_buy, market_data)
        
        return total_cost
    
    def calculate_slippage(self, price: float, quantity: int, is_buy: bool, market_data: Optional[Dict]) -> float:
        """成交量加权滑点"""
        if market_data is None or 'volume' not in market_data:
            return price * quantity * self.slippage_beta
        
        market_volume = market_data.get('volume', 1)
        order_ratio = min(quantity / market_volume, 0.1)
        
        slippage = price * quantity * (self.slippage_beta + self.slippage_alpha * order_ratio)
        return slippage
    
    def calculate_market_impact(self, price: float, quantity: int, is_buy: bool, market_data: Optional[Dict]) -> float:
        """市场冲击"""
        if market_data is None or 'volume' not in market_data:
            return 0.0
        
        market_volume = market_data.get('volume', 1)
        daily_volume = market_data.get('daily_volume', market_volume * 240)
        
        order_ratio = quantity / daily_volume
        
        impact = self.permanent_impact * np.sqrt(order_ratio) + self.temporary_impact * order_ratio
        
        return price * quantity * impact
    
    def get_cost_breakdown(
        self,
        price: float,
        quantity: int,
        is_buy: bool,
        market_data: Optional[Dict] = None
    ) -> Dict[str, float]:
        """获取成本明细"""
        breakdown = {
            'commission': self.calculate_commission(price, quantity),
            'stamp_duty': self.calculate_stamp_duty(price, quantity) if not is_buy else 0.0,
            'slippage': self.calculate_slippage(price, quantity, is_buy, market_data),
            'market_impact': self.calculate_market_impact(price, quantity, is_buy, market_data),
            'total': 0.0
        }
        
        breakdown['total'] = sum(breakdown.values())
        
        return breakdown


class ShortSellingCostModel(TransactionCostModel):
    """
    卖空成本模型
    
    卖空特有的成本:
    - 融券利息
    - 卖空佣金
    - 印花税
    - 市场冲击
    """

    def __init__(
        self,
        commission_rate: float = 0.0003,
        stamp_duty_rate: float = 0.001,
        min_commission: float = 5.0,
        short_interest_rate: float = 0.08,
        borrow_fee_rate: float = 0.0
    ):
        super().__init__()
        self.commission_rate = commission_rate
        self.stamp_duty_rate = stamp_duty_rate
        self.min_commission = min_commission
        self.short_interest_rate = short_interest_rate
        self.borrow_fee_rate = borrow_fee_rate
    
    def calculate_cost(
        self,
        price: float,
        quantity: int,
        is_buy: bool,
        market_data: Optional[Dict] = None,
        holding_days: int = 1
    ) -> float:
        """计算卖空成本"""
        total_cost = 0.0
        
        total_cost += self.calculate_commission(price, quantity)
        
        if not is_buy:
            total_cost += self.calculate_stamp_duty(price, quantity)
            total_cost += self.calculate_short_interest(price, quantity, holding_days)
        
        total_cost += self.calculate_slippage(price, quantity, is_buy, market_data)
        
        total_cost += self.calculate_market_impact(price, quantity, is_buy, market_data)
        
        return total_cost
    
    def calculate_short_interest(self, price: float, quantity: int, holding_days: int) -> float:
        """计算融券利息"""
        value = price * quantity
        daily_interest = value * (self.short_interest_rate + self.borrow_fee_rate) / 365
        return daily_interest * holding_days


class CostModelFactory:
    """
    成本模型工厂
    
    根据配置创建不同类型的成本模型
    """

    @staticmethod
    def create(model_type: str = 'comprehensive', **kwargs) -> TransactionCostModel:
        """创建成本模型"""
        if model_type == 'fixed':
            return FixedSlippageModel(**kwargs)
        elif model_type == 'volume_weighted':
            return VolumeWeightedSlippageModel(**kwargs)
        elif model_type == 'market_impact':
            return MarketImpactModel(**kwargs)
        elif model_type == 'spread':
            return SpreadCostModel(**kwargs)
        elif model_type == 'short_selling':
            return ShortSellingCostModel(**kwargs)
        else:
            return ComprehensiveCostModel(**kwargs)