import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from ..base import MultiAssetStrategy


class MLStockPickingStrategy(MultiAssetStrategy):
    def __init__(self, parameters: Dict[str, Any] = None):
        super().__init__(parameters)
        self.model_type = self.parameters.get('model_type', 'xgboost')
        self.lookback_days = self.parameters.get('lookback_days', 60)
        self.forecast_days = self.parameters.get('forecast_days', 20)
        self.top_n = self.parameters.get('top_n', 10)
        self.model = None
        self.scaler = StandardScaler()
        
    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术特征"""
        features = pd.DataFrame(index=df.index)
        
        features['return_1d'] = df['close'].pct_change(1)
        features['return_5d'] = df['close'].pct_change(5)
        features['return_20d'] = df['close'].pct_change(20)
        
        features['volatility_20d'] = df['close'].pct_change().rolling(20).std()
        
        features['ma20'] = df['close'].rolling(20).mean()
        features['ma60'] = df['close'].rolling(60).mean()
        features['ma_ratio'] = features['ma20'] / features['ma60']
        
        features['rsi'] = self._compute_rsi(df['close'], 14)
        
        macd, signal, _ = self._compute_macd(df['close'])
        features['macd'] = macd
        features['macd_signal'] = signal
        
        features['volume_ratio'] = df['volume'].rolling(20).mean() / df['volume']
        
        features = features.fillna(0)
        return features
    
    def _compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _compute_macd(self, prices: pd.Series) -> tuple:
        """计算MACD指标"""
        ema12 = prices.ewm(span=12, adjust=False).mean()
        ema26 = prices.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        return macd, signal, histogram
    
    def _create_target(self, df: pd.DataFrame) -> pd.Series:
        """创建预测目标：未来收益"""
        future_return = df['close'].pct_change(self.forecast_days).shift(-self.forecast_days)
        target = (future_return > 0.05).astype(int)
        return target
    
    def train_model(self, data: Dict[str, pd.DataFrame]):
        """训练机器学习模型"""
        all_features = []
        all_targets = []
        
        for symbol, df in data.items():
            features = self._compute_features(df)
            target = self._create_target(df)
            
            combined = pd.concat([features, target.rename('target')], axis=1)
            combined = combined.dropna()
            
            all_features.append(features.loc[combined.index])
            all_targets.append(target.loc[combined.index])
        
        X = pd.concat(all_features)
        y = pd.concat(all_targets)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        if self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        
        self.model.fit(X_train_scaled, y_train)
        
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model trained with accuracy: {accuracy:.2f}")
    
    def predict_stocks(self, data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """预测股票收益概率"""
        predictions = {}
        
        for symbol, df in data.items():
            if len(df) < self.lookback_days:
                continue
            
            features = self._compute_features(df).iloc[-1:]
            features_scaled = self.scaler.transform(features)
            prob = self.model.predict_proba(features_scaled)[0][1]
            predictions[symbol] = prob
        
        return predictions
    
    def select_stocks(self, data: Dict[str, pd.DataFrame]) -> List[str]:
        """选择股票"""
        predictions = self.predict_stocks(data)
        sorted_stocks = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return [stock[0] for stock in sorted_stocks[:self.top_n]]
    
    def get_target_weights(self, selected_stocks: List[str]) -> Dict[str, float]:
        """获取目标权重"""
        n = len(selected_stocks)
        return {stock: 1.0 / n for stock in selected_stocks}
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0
        return signals
    
    def generate_signal(self, data: pd.DataFrame) -> int:
        return 0
    
    def get_position_size(self, price: float, capital: float) -> int:
        return int(capital * 0.1 / price)
