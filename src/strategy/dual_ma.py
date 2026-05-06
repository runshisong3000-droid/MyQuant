import pandas as pd
import numpy as np
from .base import Strategy
from typing import Dict, Any


class DualMA(Strategy):
    def __init__(self, parameters: Dict[str, Any] = None):
        super().__init__(parameters)
        self.short_window = self.parameters.get('short_window', 20)
        self.long_window = self.parameters.get('long_window', 60)
        self.position_value = self.parameters.get('position_value', 100000)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0
        
        signals['short_ma'] = data['close'].rolling(window=self.short_window).mean()
        signals['long_ma'] = data['close'].rolling(window=self.long_window).mean()
        
        signals['signal'] = np.where(
            signals['short_ma'] > signals['long_ma'],
            1.0,
            0.0
        )
        
        signals['signal'] = np.where(
            signals['short_ma'] < signals['long_ma'],
            -1.0,
            signals['signal']
        )
        
        signals['signal'] = signals['signal'].diff().fillna(0)
        
        return signals
    
    def get_position_size(self, price: float, capital: float) -> int:
        max_shares = int(capital / price)
        target_shares = int(self.position_value / price)
        
        return min(max_shares, target_shares)