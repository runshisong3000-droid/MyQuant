"""
Ranking Model - 排序模型

核心功能:
    - 预测股票间的相对强弱
    - 截面收益预测
    - 生成AI评分
    - 支持多模型集成

这是中国量化私募的核心方法论：
    不是预测涨跌，而是预测相对强弱
    然后：买前10%，卖后10%，做中性对冲
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime


class CrossSectionalRankingLoss(nn.Module):
    """截面排序损失函数"""

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, scores: torch.Tensor, returns: torch.Tensor):
        """
        Args:
            scores: 模型预测分数 [batch_size]
            returns: 实际收益率 [batch_size]
        """
        n = scores.size(0)
        
        scores_diff = scores.unsqueeze(0) - scores.unsqueeze(1)
        returns_diff = returns.unsqueeze(0) - returns.unsqueeze(1)
        
        mask = (returns_diff > 0).float()
        
        loss = torch.mean(
            torch.max(
                self.margin - mask * scores_diff,
                torch.zeros_like(scores_diff)
            )
        )
        
        return loss


class RankingNet(nn.Module):
    """排序神经网络"""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.2
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x).squeeze()


class CrossSectionalDataLoader:
    """截面数据加载器"""

    def __init__(
        self,
        features: pd.DataFrame,
        returns: pd.Series,
        lookahead_periods: int = 1,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ):
        self.features = features
        self.returns = returns
        self.lookahead_periods = lookahead_periods
        
        self.dates = sorted(features.index.get_level_values(0).unique())
        n_train = int(len(self.dates) * train_ratio)
        n_val = int(len(self.dates) * val_ratio)
        
        self.train_dates = self.dates[:n_train]
        self.val_dates = self.dates[n_train:n_train + n_val]
        self.test_dates = self.dates[n_train + n_val:]
        
    def get_data_by_date(self, date):
        """获取指定日期的数据"""
        X = self.features.loc[date].values
        y = self.returns.loc[date].values
        
        valid_mask = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
        
        return X[valid_mask], y[valid_mask]
    
    def iter_dates(self, split: str = 'train'):
        """迭代日期"""
        if split == 'train':
            dates = self.train_dates
        elif split == 'val':
            dates = self.val_dates
        else:
            dates = self.test_dates
            
        for date in dates[:-self.lookahead_periods]:
            X, _ = self.get_data_by_date(date)
            next_date = self.dates[self.dates.index(date) + self.lookahead_periods]
            _, y = self.get_data_by_date(next_date)
            
            if len(X) > 0 and len(y) > 0:
                yield X, y


class RankingModel(BaseEstimator, TransformerMixin):
    """
    排序模型 - 预测截面相对收益
    
    核心思想:
        不是预测涨跌，而是预测股票间的相对强弱
        输出: AI评分，用于排序选股
    """

    def __init__(
        self,
        model_type: str = 'lgbm',
        params: Optional[Dict] = None,
        use_gpu: bool = False
    ):
        self.model_type = model_type
        self.params = params or {}
        self.use_gpu = use_gpu
        self.model = None
        self.scaler = StandardScaler()
        
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dates: Optional[np.ndarray] = None
    ):
        """
        训练模型
        
        Args:
            X: 特征矩阵 [n_samples, n_features]
            y: 目标收益率 [n_samples]
            dates: 日期标签，用于分组
        """
        X = self.scaler.fit_transform(X)
        
        if self.model_type == 'lgbm':
            params = {
                'objective': 'regression',
                'metric': 'mse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbosity': -1,
                **self.params
            }
            
            if self.use_gpu:
                params['device'] = 'gpu'
                params['gpu_platform_id'] = 0
                params['gpu_device_id'] = 0
            
            if dates is not None:
                train_data = lgb.Dataset(X, label=y, group=dates)
            else:
                train_data = lgb.Dataset(X, label=y)
            
            self.model = lgb.train(params, train_data, num_boost_round=100)
        
        elif self.model_type == 'nn':
            self.model = RankingNet(input_dim=X.shape[1])
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            criterion = CrossSectionalRankingLoss()
            
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32)
            
            for epoch in range(50):
                self.model.train()
                optimizer.zero_grad()
                
                scores = self.model(X_tensor)
                loss = criterion(scores, y_tensor)
                
                loss.backward()
                optimizer.step()
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        elif self.model_type == 'linear':
            from sklearn.linear_model import LinearRegression
            self.model = LinearRegression(**self.params)
            self.model.fit(X, y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测评分"""
        X = self.scaler.transform(X)
        
        if self.model_type == 'lgbm':
            return self.model.predict(X)
        elif self.model_type == 'nn':
            self.model.eval()
            X_tensor = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                return self.model(X_tensor).cpu().numpy()
        elif self.model_type == 'linear':
            return self.model.predict(X)
    
    def predict_rank(self, X: np.ndarray) -> np.ndarray:
        """预测排名"""
        scores = self.predict(X)
        ranks = np.argsort(np.argsort(-scores)) + 1
        return ranks
    
    def evaluate_ic(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算IC (Information Coefficient)"""
        scores = self.predict(X)
        return np.corrcoef(scores, y)[0, 1]
    
    def evaluate_ic_ir(self, X: np.ndarray, y: np.ndarray, dates: np.ndarray) -> Tuple[float, float]:
        """计算IC和IR (Information Ratio)"""
        unique_dates = np.unique(dates)
        ics = []
        
        for date in unique_dates:
            mask = dates == date
            if np.sum(mask) > 1:
                ic = np.corrcoef(self.predict(X[mask]), y[mask])[0, 1]
                if not np.isnan(ic):
                    ics.append(ic)
        
        ic_mean = np.mean(ics)
        ir = ic_mean / np.std(ics)
        
        return ic_mean, ir
    
    def save_model(self, path: str):
        """保存模型"""
        import joblib
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'params': self.params
        }, path)
    
    @classmethod
    def load_model(cls, path: str):
        """加载模型"""
        import joblib
        data = joblib.load(path)
        model = cls(model_type=data['model_type'], params=data['params'])
        model.model = data['model']
        model.scaler = data['scaler']
        return model


class EnsembleRankingModel:
    """集成排序模型"""

    def __init__(self, models: List[RankingModel], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """集成预测"""
        predictions = np.zeros((len(self.models), len(X)))
        
        for i, model in enumerate(self.models):
            predictions[i] = model.predict(X)
        
        return np.average(predictions, axis=0, weights=self.weights)
    
    def predict_rank(self, X: np.ndarray) -> np.ndarray:
        """集成排名"""
        scores = self.predict(X)
        ranks = np.argsort(np.argsort(-scores)) + 1
        return ranks
    
    def evaluate_ic(self, X: np.ndarray, y: np.ndarray) -> float:
        """评估IC"""
        scores = self.predict(X)
        return np.corrcoef(scores, y)[0, 1]
    
    def save_model(self, path: str):
        """保存模型"""
        import joblib
        joblib.dump({
            'models': self.models,
            'weights': self.weights
        }, path)
    
    @classmethod
    def load_model(cls, path: str):
        """加载模型"""
        import joblib
        data = joblib.load(path)
        return cls(models=data['models'], weights=data['weights'])


class CrossSectionalTrainer:
    """截面训练器"""

    def __init__(self, model: RankingModel, lookahead_periods: int = 1):
        self.model = model
        self.lookahead_periods = lookahead_periods
        self.ic_history = []
        self.ir_history = []
        
    def train(self, data_loader: CrossSectionalDataLoader, epochs: int = 1):
        """训练模型"""
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            all_X, all_y = [], []
            
            for X, y in data_loader.iter_dates('train'):
                all_X.append(X)
                all_y.append(y)
            
            if all_X:
                X_train = np.vstack(all_X)
                y_train = np.concatenate(all_y)
                
                self.model.fit(X_train, y_train)
                
                val_ic = self.evaluate(data_loader, 'val')
                print(f"Validation IC: {val_ic:.4f}")
                
                self.ic_history.append(val_ic)
    
    def evaluate(self, data_loader: CrossSectionalDataLoader, split: str = 'val') -> float:
        """评估模型"""
        all_scores, all_returns = [], []
        
        for X, y in data_loader.iter_dates(split):
            scores = self.model.predict(X)
            all_scores.extend(scores)
            all_returns.extend(y)
        
        if len(all_scores) < 2:
            return 0.0
        
        return np.corrcoef(all_scores, all_returns)[0, 1]
    
    def generate_scores(self, features: pd.DataFrame) -> pd.Series:
        """生成评分"""
        scores = {}
        
        for date in features.index.get_level_values(0).unique():
            X = features.loc[date].values
            valid_mask = ~np.any(np.isnan(X), axis=1)
            
            if np.sum(valid_mask) > 0:
                date_scores = np.full(len(valid_mask), np.nan)
                date_scores[valid_mask] = self.model.predict(X[valid_mask])
                stocks = features.loc[date].index[valid_mask]
                for stock, score in zip(stocks, date_scores[valid_mask]):
                    scores[(date, stock)] = score
        
        return pd.Series(scores, name='ai_score')