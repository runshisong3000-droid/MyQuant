import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List
from collections import defaultdict


class BacktestEngine:
    def __init__(self, config_path: str = "config/backtest.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.mode = self.config.get('mode', 'event_driven')
        self.timezone = self.config.get('timezone', 'Asia/Shanghai')
        
        self.results = None
        self.trades = []
        self.daily_equity = []
        
    def run_event_driven(self, strategy, data: pd.DataFrame, 
                        initial_capital: float = 1000000.0) -> pd.DataFrame:
        """
        事件驱动回测
        
        参数:
            strategy: 策略对象
            data: 价格数据 DataFrame
            initial_capital: 初始资金
        
        返回:
            回测结果 DataFrame
        """
        portfolio = pd.DataFrame(index=data.index)
        portfolio['price'] = data['close']
        portfolio['cash'] = initial_capital
        portfolio['holdings'] = 0.0
        portfolio['total'] = initial_capital
        portfolio['position'] = 0
        
        cash = initial_capital
        holdings = 0
        position = 0
        
        for i, date in enumerate(data.index):
            price = data.loc[date, 'close']
            
            signal = strategy.generate_signal(data.iloc[:i+1])
            
            if signal == 1 and cash > 0:
                shares = strategy.get_position_size(price, cash)
                cost = shares * price * (1 + self.config['event_driven'].get('transaction_cost', 0))
                if cost <= cash:
                    holdings += shares
                    cash -= cost
                    position = shares
                    self._record_trade(date, 'BUY', price, shares, cost)
            
            elif signal == -1 and holdings > 0:
                revenue = holdings * price * (1 - self.config['event_driven'].get('transaction_cost', 0))
                cash += revenue
                self._record_trade(date, 'SELL', price, holdings, revenue)
                holdings = 0
                position = 0
            
            portfolio.loc[date, 'cash'] = cash
            portfolio.loc[date, 'holdings'] = holdings * price
            portfolio.loc[date, 'total'] = cash + holdings * price
            portfolio.loc[date, 'position'] = position
            portfolio.loc[date, 'signal'] = signal
        
        self.results = portfolio
        return portfolio
    
    def run_vectorized(self, strategy, data: pd.DataFrame,
                      initial_capital: float = 1000000.0) -> pd.DataFrame:
        """
        向量化回测（快速版本）
        
        参数:
            strategy: 策略对象
            data: 价格数据 DataFrame
            initial_capital: 初始资金
        
        返回:
            回测结果 DataFrame
        """
        signals = strategy.generate_signals(data)
        positions = signals['signal'].shift().fillna(0)
        
        portfolio = pd.DataFrame(index=data.index)
        portfolio['price'] = data['close']
        portfolio['signal'] = signals['signal']
        portfolio['position'] = positions
        
        portfolio['return'] = data['close'].pct_change()
        portfolio['strategy_return'] = positions * portfolio['return']
        
        portfolio['cumulative_return'] = (1 + portfolio['strategy_return']).cumprod()
        portfolio['equity'] = initial_capital * portfolio['cumulative_return']
        
        self.results = portfolio
        return portfolio
    
    def _record_trade(self, date, action: str, price: float, 
                     shares: int, amount: float):
        """记录交易"""
        self.trades.append({
            'date': date,
            'action': action,
            'price': price,
            'shares': shares,
            'amount': amount
        })
    
    def run(self, strategy, data: pd.DataFrame, 
           initial_capital: float = 1000000.0) -> pd.DataFrame:
        """
        运行回测
        
        参数:
            strategy: 策略对象
            data: 价格数据 DataFrame
            initial_capital: 初始资金
        
        返回:
            回测结果 DataFrame
        """
        if self.mode == 'event_driven':
            return self.run_event_driven(strategy, data, initial_capital)
        else:
            return self.run_vectorized(strategy, data, initial_capital)
    
    def get_trades(self) -> pd.DataFrame:
        """获取交易记录"""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(self.trades)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取回测摘要"""
        if self.results is None:
            return {}
        
        total_return = (self.results['total'].iloc[-1] / self.results['total'].iloc[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(self.results)) - 1
        
        returns = self.results['total'].pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        
        portfolio_value = self.results['total']
        max_dd = (portfolio_value / portfolio_value.cummax() - 1).min()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': abs(max_dd),
            'num_trades': len(self.trades),
            'start_date': self.results.index[0],
            'end_date': self.results.index[-1]
        }
