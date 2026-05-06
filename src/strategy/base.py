from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional


class Strategy(ABC):
    def __init__(self, parameters: Dict[str, Any] = None):
        self.parameters = parameters or {}
        self.signals = pd.DataFrame()
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def get_position_size(self, price: float, capital: float) -> int:
        pass
    
    def run(self, data: pd.DataFrame, initial_capital: float = 1000000.0) -> pd.DataFrame:
        self.signals = self.generate_signals(data)
        return self.backtest(data, initial_capital)
    
    def backtest(self, data: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
        portfolio = pd.DataFrame(index=data.index)
        portfolio['price'] = data['close']
        portfolio['signal'] = self.signals['signal']
        
        portfolio['position'] = portfolio['signal'].shift(1).fillna(0)
        portfolio['position'] = portfolio['position'].astype(int)
        
        portfolio['cash'] = initial_capital
        portfolio['holdings'] = 0.0
        portfolio['total'] = initial_capital
        
        cash = initial_capital
        holdings = 0
        
        for i in range(1, len(portfolio)):
            date = portfolio.index[i]
            prev_date = portfolio.index[i-1]
            signal = portfolio.loc[prev_date, 'signal']
            price = portfolio.loc[date, 'price']
            
            if signal == 1 and cash > 0:
                position_size = self.get_position_size(price, cash)
                cost = position_size * price
                if cost <= cash:
                    holdings += position_size
                    cash -= cost
            
            elif signal == -1 and holdings > 0:
                cash += holdings * price
                holdings = 0
            
            portfolio.loc[date, 'cash'] = cash
            portfolio.loc[date, 'holdings'] = holdings * price
            portfolio.loc[date, 'total'] = cash + holdings * price
        
        return portfolio