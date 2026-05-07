"""
模型集成模块

功能:
    - Stacking集成
    - Blending集成
    - 投票集成
    - 模型融合
"""

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from typing import List, Dict, Union, Optional
import joblib


class EnsembleModel:
    """集成模型基类"""

    def __init__(self, models: List, ensemble_type: str = 'voting'):
        self.models = models
        self.ensemble_type = ensemble_type
        self.meta_model = None
        self.is_classifier = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """训练模型"""
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        pass

    def save_model(self, path: str):
        """保存模型"""
        joblib.dump({
            'models': self.models,
            'ensemble_type': self.ensemble_type,
            'meta_model': self.meta_model,
            'is_classifier': self.is_classifier
        }, path)

    @classmethod
    def load_model(cls, path: str):
        """加载模型"""
        data = joblib.load(path)
        model = cls(data['models'], data['ensemble_type'])
        model.meta_model = data['meta_model']
        model.is_classifier = data['is_classifier']
        return model


class VotingEnsemble(EnsembleModel):
    """投票集成"""

    def __init__(
        self,
        models: List,
        voting: str = 'hard',
        weights: Optional[List[float]] = None
    ):
        super().__init__(models, 'voting')
        self.voting = voting
        self.weights = weights
        self.ensemble = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.is_classifier = len(np.unique(y)) <= 10

        if self.is_classifier:
            self.ensemble = VotingClassifier(
                estimators=[(f'model_{i}', m) for i, m in enumerate(self.models)],
                voting=self.voting,
                weights=self.weights
            )
        else:
            self.ensemble = VotingRegressor(
                estimators=[(f'model_{i}', m) for i, m in enumerate(self.models)],
                weights=self.weights
            )

        self.ensemble.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.ensemble.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.ensemble, 'predict_proba'):
            return self.ensemble.predict_proba(X)
        return self.predict(X)


class StackingEnsemble(EnsembleModel):
    """Stacking集成"""

    def __init__(
        self,
        base_models: List,
        meta_model=None,
        n_folds: int = 5
    ):
        super().__init__(base_models, 'stacking')
        self.n_folds = n_folds
        if meta_model is None:
            self.meta_model = LogisticRegression()
        else:
            self.meta_model = meta_model
        self.fitted_models = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.is_classifier = len(np.unique(y)) <= 10

        n_samples = X.shape[0]
        meta_features = np.zeros((n_samples, len(self.models)))

        for model_idx, model in enumerate(self.models):
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train = y[train_idx]

                model.fit(X_train, y_train)
                meta_features[val_idx, model_idx] = model.predict(X_val)

            model.fit(X, y)
            self.fitted_models.append(model)

        self.meta_model.fit(meta_features, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        meta_features = np.zeros((X.shape[0], len(self.fitted_models)))

        for model_idx, model in enumerate(self.fitted_models):
            meta_features[:, model_idx] = model.predict(X)

        return self.meta_model.predict(meta_features)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        meta_features = np.zeros((X.shape[0], len(self.fitted_models)))

        for model_idx, model in enumerate(self.fitted_models):
            if hasattr(model, 'predict_proba'):
                meta_features[:, model_idx] = model.predict_proba(X)[:, 1]
            else:
                meta_features[:, model_idx] = model.predict(X)

        if hasattr(self.meta_model, 'predict_proba'):
            return self.meta_model.predict_proba(meta_features)
        return self.meta_model.predict(meta_features)


class BlendingEnsemble(EnsembleModel):
    """Blending集成"""

    def __init__(
        self,
        base_models: List,
        meta_model=None,
        val_size: float = 0.2
    ):
        super().__init__(base_models, 'blending')
        self.val_size = val_size
        if meta_model is None:
            self.meta_model = LogisticRegression()
        else:
            self.meta_model = meta_model

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.is_classifier = len(np.unique(y)) <= 10

        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.val_size, random_state=42
        )

        meta_features = np.zeros((len(y_val), len(self.models)))

        for i, model in enumerate(self.models):
            model.fit(X_train, y_train)
            meta_features[:, i] = model.predict(X_val)

        self.meta_model.fit(meta_features, y_val)

        for model in self.models:
            model.fit(X, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        meta_features = np.zeros((X.shape[0], len(self.models)))

        for i, model in enumerate(self.models):
            meta_features[:, i] = model.predict(X)

        return self.meta_model.predict(meta_features)


class WeightedEnsemble(EnsembleModel):
    """加权集成"""

    def __init__(
        self,
        models: List,
        weights: Optional[List[float]] = None
    ):
        super().__init__(models, 'weighted')
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            self.weights = weights

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.is_classifier = len(np.unique(y)) <= 10

        for model in self.models:
            model.fit(X, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = []

        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)

        predictions = np.array(predictions)

        if self.is_classifier:
            weighted_preds = np.average(predictions, axis=0, weights=self.weights)
            return np.round(weighted_preds).astype(int)
        else:
            return np.average(predictions, axis=0, weights=self.weights)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probas = []

        for model in self.models:
            if hasattr(model, 'predict_proba'):
                probas.append(model.predict_proba(X))
            else:
                probas.append(model.predict(X).reshape(-1, 1))

        probas = np.array(probas)
        return np.average(probas, axis=0, weights=self.weights)


class ModelEnsembleFactory:
    """模型集成工厂"""

    @staticmethod
    def create_classifier_ensemble(
        ensemble_type: str = 'voting',
        models: Optional[List] = None,
        **kwargs
    ) -> EnsembleModel:
        """创建分类器集成"""
        if models is None:
            models = [
                XGBClassifier(n_estimators=100, random_state=42),
                LGBMClassifier(n_estimators=100, random_state=42),
                RandomForestClassifier(n_estimators=100, random_state=42),
                SVC(probability=True, random_state=42)
            ]

        if ensemble_type == 'voting':
            return VotingEnsemble(models, voting=kwargs.get('voting', 'soft'))
        elif ensemble_type == 'stacking':
            return StackingEnsemble(models, meta_model=kwargs.get('meta_model'))
        elif ensemble_type == 'blending':
            return BlendingEnsemble(models, meta_model=kwargs.get('meta_model'))
        elif ensemble_type == 'weighted':
            return WeightedEnsemble(models, weights=kwargs.get('weights'))
        else:
            raise ValueError(f"Unknown ensemble type: {ensemble_type}")

    @staticmethod
    def create_regressor_ensemble(
        ensemble_type: str = 'voting',
        models: Optional[List] = None,
        **kwargs
    ) -> EnsembleModel:
        """创建回归器集成"""
        if models is None:
            models = [
                XGBRegressor(n_estimators=100, random_state=42),
                LGBMRegressor(n_estimators=100, random_state=42),
                RandomForestRegressor(n_estimators=100, random_state=42),
                SVR()
            ]

        if ensemble_type == 'voting':
            return VotingEnsemble(models, voting='hard')
        elif ensemble_type == 'stacking':
            meta_model = kwargs.get('meta_model', LinearRegression())
            return StackingEnsemble(models, meta_model=meta_model)
        elif ensemble_type == 'blending':
            meta_model = kwargs.get('meta_model', LinearRegression())
            return BlendingEnsemble(models, meta_model=meta_model)
        elif ensemble_type == 'weighted':
            return WeightedEnsemble(models, weights=kwargs.get('weights'))
        else:
            raise ValueError(f"Unknown ensemble type: {ensemble_type}")


class ModelSelector:
    """模型选择器"""

    def __init__(self):
        self.models = {
            'classifier': {
                'lr': LogisticRegression,
                'dt': DecisionTreeClassifier,
                'rf': RandomForestClassifier,
                'xgb': XGBClassifier,
                'lgbm': LGBMClassifier,
                'svc': SVC
            },
            'regressor': {
                'lr': LinearRegression,
                'dt': DecisionTreeRegressor,
                'rf': RandomForestRegressor,
                'xgb': XGBRegressor,
                'lgbm': LGBMRegressor,
                'svr': SVR
            }
        }

    def get_model(self, model_name: str, model_type: str = 'classifier', **kwargs):
        """获取模型"""
        if model_type not in self.models:
            raise ValueError(f"Unknown model type: {model_type}")

        if model_name not in self.models[model_type]:
            raise ValueError(f"Unknown model: {model_name}")

        model_class = self.models[model_type][model_name]
        return model_class(**kwargs)

    def list_models(self, model_type: str = None):
        """列出可用模型"""
        if model_type is None:
            return {k: list(v.keys()) for k, v in self.models.items()}
        return list(self.models.get(model_type, {}).keys())