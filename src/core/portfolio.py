import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from collections import defaultdict


class Portfolio:
    def __init__(self, initial_capital: float = 1000000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = defaultdict(int)
        self.holdings = {}
        self.trades = []
        
    def update_positions(self, date: pd.Timestamp, prices: Dict[str, float]):
        """更新持仓价值"""
        total_holdings = 0.0
        for symbol, shares in self.positions.items():
            if symbol in prices:
                self.holdings[symbol] = shares * prices[symbol]
                total_holdings += self.holdings[symbol]
        return total_holdings
    
    def get_total_value(self, prices: Dict[str, float]) -> float:
        """获取总市值"""
        holdings_value = self.update_positions(None, prices)
        return self.cash + holdings_value
    
    def buy(self, date: pd.Timestamp, symbol: str, price: float, 
           shares: int, transaction_cost: float = 0.001):
        """买入股票"""
        cost = shares * price * (1 + transaction_cost)
        if cost <= self.cash:
            self.positions[symbol] += shares
            self.cash -= cost
            self.trades.append({
                'date': date,
                'action': 'BUY',
                'symbol': symbol,
                'price': price,
                'shares': shares,
                'cost': cost
            })
            return True
        return False
    
    def sell(self, date: pd.Timestamp, symbol: str, price: float,
            shares: Optional[int] = None, transaction_cost: float = 0.001):
        """卖出股票"""
        if shares is None:
            shares = self.positions[symbol]
        
        if self.positions[symbol] >= shares:
            revenue = shares * price * (1 - transaction_cost)
            self.positions[symbol] -= shares
            self.cash += revenue
            self.trades.append({
                'date': date,
                'action': 'SELL',
                'symbol': symbol,
                'price': price,
                'shares': shares,
                'revenue': revenue
            })
            return True
        return False
    
    def rebalance(self, date: pd.Timestamp, target_weights: Dict[str, float],
                 prices: Dict[str, float], transaction_cost: float = 0.001):
        """再平衡组合"""
        total_value = self.get_total_value(prices)
        
        for symbol, target_weight in target_weights.items():
            target_value = total_value * target_weight
            current_shares = self.positions.get(symbol, 0)
            current_value = current_shares * prices.get(symbol, 0)
            
            diff_value = target_value - current_value
            
            if diff_value > 0:
                shares_to_buy = int(diff_value / prices[symbol])
                if shares_to_buy > 0:
                    self.buy(date, symbol, prices[symbol], shares_to_buy, transaction_cost)
            elif diff_value < 0:
                shares_to_sell = int(-diff_value / prices[symbol])
                if shares_to_sell > 0:
                    self.sell(date, symbol, prices[symbol], shares_to_sell, transaction_cost)
    
    def get_positions_df(self) -> pd.DataFrame:
        """获取持仓DataFrame"""
        positions_data = []
        for symbol, shares in self.positions.items():
            positions_data.append({
                'symbol': symbol,
                'shares': shares
            })
        return pd.DataFrame(positions_data)
    
    def get_trades_df(self) -> pd.DataFrame:
        """获取交易记录DataFrame"""
        return pd.DataFrame(self.trades)
    
    def clear(self):
        """清空持仓"""
        self.positions.clear()
        self.holdings.clear()
        self.cash = self.initial_capital
        self.trades = []


class MultiAssetPortfolio(Portfolio):
    def __init__(self, initial_capital: float = 1000000.0, max_stocks: int = 10):
        super().__init__(initial_capital)
        self.max_stocks = max_stocks
        self.daily_values = []
    
    def update_daily_value(self, date: pd.Timestamp, prices: Dict[str, float]):
        """记录每日市值"""
        total_value = self.get_total_value(prices)
        self.daily_values.append({
            'date': date,
            'total': total_value,
            'cash': self.cash,
            'holdings': total_value - self.cash
        })
    
    def get_equity_curve(self) -> pd.DataFrame:
        """获取权益曲线"""
        df = pd.DataFrame(self.daily_values)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df['return'] = df['total'].pct_change()
        df['cumulative_return'] = (1 + df['return']).cumprod() - 1
        return df
    
    def get_weights(self, prices: Dict[str, float]) -> Dict[str, float]:
        """获取当前权重"""
        total_value = self.get_total_value(prices)
        weights = {}
        for symbol, shares in self.positions.items():
            if symbol in prices:
                weights[symbol] = (shares * prices[symbol]) / total_value
        return weights
