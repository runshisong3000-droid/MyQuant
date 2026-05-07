import pandas as pd
import numpy as np
from typing import List, Dict, Optional


class FeatureEngineer:
    def __init__(self):
        pass
    
    def add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加收益特征"""
        df = df.copy()
        
        periods = [1, 5, 10, 20, 60]
        for period in periods:
            df[f'return_{period}d'] = df['close'].pct_change(period)
        
        return df
    
    def add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加均线特征"""
        df = df.copy()
        
        windows = [10, 20, 50, 100, 200]
        for window in windows:
            df[f'ma_{window}'] = df['close'].rolling(window).mean()
            df[f'ma_ratio_{window}'] = df['close'] / df[f'ma_{window}']
        
        return df
    
    def add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加波动率特征"""
        df = df.copy()
        
        df['volatility_20d'] = df['close'].pct_change().rolling(20).std()
        df['volatility_60d'] = df['close'].pct_change().rolling(60).std()
        df['volatility_ratio'] = df['volatility_20d'] / df['volatility_60d']
        
        return df
    
    def add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """添加RSI指标"""
        df = df.copy()
        
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加MACD指标"""
        df = df.copy()
        
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        return df
    
    def add_bollinger_bands(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """添加布林带"""
        df = df.copy()
        
        df['bb_mid'] = df['close'].rolling(window).mean()
        df['bb_std'] = df['close'].rolling(window).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
        
        return df
    
    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加成交量特征"""
        df = df.copy()
        
        df['volume_ma20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma20']
        df['volume_change'] = df['volume'].pct_change(5)
        
        return df
    
    def add_drawdown(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加回撤特征"""
        df = df.copy()
        
        df['peak'] = df['close'].rolling(60).max()
        df['drawdown'] = (df['close'] / df['peak']) - 1
        df['max_drawdown'] = df['drawdown'].rolling(60).min()
        
        return df
    
    def generate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成所有特征"""
        df = self.add_returns(df)
        df = self.add_moving_averages(df)
        df = self.add_volatility(df)
        df = self.add_rsi(df)
        df = self.add_macd(df)
        df = self.add_bollinger_bands(df)
        df = self.add_volume_features(df)
        df = self.add_drawdown(df)
        
        return df.dropna()
