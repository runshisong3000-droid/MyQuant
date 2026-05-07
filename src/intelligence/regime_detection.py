import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from typing import Optional, List


class RegimeDetector:
    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.model = KMeans(n_clusters=n_regimes, random_state=42)
        self.pca = PCA(n_components=2)
        
    def compute_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算用于识别市场状态的特征"""
        features = pd.DataFrame(index=data.index)
        
        features['return_1d'] = data['close'].pct_change(1)
        features['return_5d'] = data['close'].pct_change(5)
        features['return_20d'] = data['close'].pct_change(20)
        
        features['volatility_20d'] = features['return_1d'].rolling(20).std()
        features['volatility_60d'] = features['return_1d'].rolling(60).std()
        
        features['volume_change'] = data['volume'].pct_change(5)
        
        features['momentum'] = features['return_20d'] / features['volatility_20d']
        
        features['drawdown'] = (data['close'] / data['close'].rolling(60).max()) - 1
        
        features = features.fillna(0)
        return features
    
    def fit(self, data: pd.DataFrame):
        """训练市场状态识别模型"""
        features = self.compute_regime_features(data)
        features_pca = self.pca.fit_transform(features)
        self.model.fit(features_pca)
        
        explained_variance = self.pca.explained_variance_ratio_.sum()
        print(f"PCA explained variance: {explained_variance:.2f}")
    
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """预测市场状态"""
        features = self.compute_regime_features(data)
        features_pca = self.pca.transform(features)
        regimes = self.model.predict(features_pca)
        return pd.Series(regimes, index=data.index, name='regime')
    
    def analyze_regimes(self, data: pd.DataFrame) -> dict:
        """分析各状态特征"""
        features = self.compute_regime_features(data)
        regimes = self.predict(data)
        
        analysis = {}
        for regime in range(self.n_regimes):
            mask = regimes == regime
            regime_features = features[mask]
            
            analysis[regime] = {
                'count': mask.sum(),
                'avg_return': regime_features['return_1d'].mean(),
                'avg_volatility': regime_features['volatility_20d'].mean(),
                'avg_drawdown': regime_features['drawdown'].mean()
            }
        
        return analysis


class VolatilityRegimeDetector:
    def __init__(self, window: int = 20, threshold_multiplier: float = 2.0):
        self.window = window
        self.threshold_multiplier = threshold_multiplier
        
    def detect(self, data: pd.DataFrame) -> pd.Series:
        """检测波动率状态"""
        returns = data['close'].pct_change()
        volatility = returns.rolling(self.window).std()
        
        avg_volatility = volatility.mean()
        threshold = avg_volatility * self.threshold_multiplier
        
        regimes = pd.Series(0, index=data.index)
        regimes[volatility > threshold] = 1
        
        return regimes.rename('volatility_regime')
