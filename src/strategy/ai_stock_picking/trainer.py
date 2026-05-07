"""
模型训练和验证框架

功能:
    - 数据预处理
    - 交叉验证
    - 超参数调优
    - 模型评估
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from typing import Dict, List, Tuple, Optional, Any
import time


class DataProcessor:
    """数据处理器"""

    def __init__(self):
        self.scaler = None

    def fit_transform(
        self,
        X: np.ndarray,
        scaler_type: str = 'standard',
        fit_y: bool = False,
        y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """拟合并转换数据"""
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

        X_scaled = self.scaler.fit_transform(X)

        if fit_y and y is not None:
            y = y.reshape(-1, 1)
            y_scaled = self.scaler.fit_transform(y).flatten()
            return X_scaled, y_scaled

        return X_scaled, None

    def transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """转换数据"""
        X_scaled = self.scaler.transform(X)

        if y is not None:
            y = y.reshape(-1, 1)
            y_scaled = self.scaler.transform(y).flatten()
            return X_scaled, y_scaled

        return X_scaled, None

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """逆转换"""
        return self.scaler.inverse_transform(y.reshape(-1, 1)).flatten()


class CrossValidator:
    """交叉验证器"""

    def __init__(self, n_folds: int = 5, stratified: bool = True):
        self.n_folds = n_folds
        self.stratified = stratified

    def split(self, X: np.ndarray, y: np.ndarray):
        """生成交叉验证分割"""
        if self.stratified and len(np.unique(y)) <= 10:
            kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        else:
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(X, y):
            yield train_idx, val_idx

    def cross_validate(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        metrics: List[str] = None
    ) -> Dict[str, List[float]]:
        """执行交叉验证"""
        if metrics is None:
            metrics = ['accuracy', 'f1']

        results = {metric: [] for metric in metrics}
        results['time'] = []

        for train_idx, val_idx in self.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            start_time = time.time()
            model.fit(X_train, y_train)
            elapsed_time = time.time() - start_time

            y_pred = model.predict(X_val)

            for metric in metrics:
                score = self._calculate_metric(y_val, y_pred, metric)
                results[metric].append(score)

            results['time'].append(elapsed_time)

        return results

    def _calculate_metric(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric: str
    ) -> float:
        """计算指标"""
        if metric == 'accuracy':
            return accuracy_score(y_true, y_pred)
        elif metric == 'precision':
            return precision_score(y_true, y_pred, average='weighted')
        elif metric == 'recall':
            return recall_score(y_true, y_pred, average='weighted')
        elif metric == 'f1':
            return f1_score(y_true, y_pred, average='weighted')
        elif metric == 'roc_auc':
            try:
                return roc_auc_score(y_true, y_pred)
            except:
                return 0.0
        elif metric == 'mse':
            return mean_squared_error(y_true, y_pred)
        elif metric == 'mae':
            return mean_absolute_error(y_true, y_pred)
        elif metric == 'r2':
            return r2_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")


class HyperparameterTuner:
    """超参数调优器"""

    def __init__(self, search_type: str = 'grid', n_iter: int = 10):
        self.search_type = search_type
        self.n_iter = n_iter
        self.best_model = None
        self.best_params = None
        self.best_score = None

    def tune(
        self,
        model,
        param_grid: Dict[str, List],
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        scoring: str = 'accuracy'
    ):
        """执行超参数调优"""
        if self.search_type == 'grid':
            search = GridSearchCV(
                model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1
            )
        elif self.search_type == 'random':
            search = RandomizedSearchCV(
                model, param_grid, n_iter=self.n_iter, cv=cv,
                scoring=scoring, n_jobs=-1, verbose=1, random_state=42
            )
        else:
            raise ValueError(f"Unknown search type: {self.search_type}")

        search.fit(X, y)

        self.best_model = search.best_estimator_
        self.best_params = search.best_params_
        self.best_score = search.best_score_

        return self.best_model, self.best_params, self.best_score


class ModelEvaluator:
    """模型评估器"""

    def __init__(self):
        pass

    def evaluate_classifier(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, float]:
        """评估分类器"""
        y_pred = model.predict(X_test)

        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }

        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
            try:
                results['roc_auc'] = roc_auc_score(y_test, y_proba)
            except:
                results['roc_auc'] = 0.0

        if verbose:
            self._print_results(results, 'Classification Metrics')

        return results

    def evaluate_regressor(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, float]:
        """评估回归器"""
        y_pred = model.predict(X_test)

        results = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }

        if verbose:
            self._print_results(results, 'Regression Metrics')

        return results

    def _print_results(self, results: Dict[str, float], title: str):
        """打印评估结果"""
        print(f"\n{title}:")
        print("-" * 40)
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
        print("-" * 40)


class FeatureSelector:
    """特征选择器"""

    def __init__(self, method: str = 'importance', threshold: float = 0.01):
        self.method = method
        self.threshold = threshold
        self.selected_features = None

    def fit(self, model, X: np.ndarray, feature_names: Optional[List[str]] = None):
        """拟合特征选择"""
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        if self.method == 'importance':
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                self.selected_features = [
                    name for name, imp in zip(feature_names, importances)
                    if imp >= self.threshold
                ]
            else:
                raise ValueError("Model doesn't have feature_importances_ attribute")

        elif self.method == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=self.threshold)
            pca.fit(X)
            self.selected_features = [
                f'pc_{i}' for i in range(pca.n_components_)
            ]

        else:
            raise ValueError(f"Unknown method: {self.method}")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """转换数据"""
        if self.method == 'importance':
            mask = self._get_feature_mask(X.shape[1])
            return X[:, mask]
        elif self.method == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=len(self.selected_features))
            return pca.fit_transform(X)

    def _get_feature_mask(self, n_features: int) -> List[bool]:
        """获取特征掩码"""
        mask = []
        for i in range(n_features):
            feature_name = f'feature_{i}'
            mask.append(feature_name in self.selected_features)
        return mask


class Pipeline:
    """机器学习管道"""

    def __init__(self, steps: List[Tuple[str, Any]]):
        self.steps = steps
        self.named_steps = {name: step for name, step in steps}

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """拟合管道"""
        for name, step in self.steps:
            if hasattr(step, 'fit_transform'):
                if y is not None:
                    X, y = step.fit_transform(X, y=y)
                else:
                    X = step.fit_transform(X)
            elif hasattr(step, 'fit'):
                step.fit(X, y)
            else:
                raise ValueError(f"Step {name} doesn't have fit or fit_transform method")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        for name, step in self.steps[:-1]:
            if hasattr(step, 'transform'):
                X = step.transform(X)

        final_step = self.steps[-1][1]
        return final_step.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        for name, step in self.steps[:-1]:
            if hasattr(step, 'transform'):
                X = step.transform(X)

        final_step = self.steps[-1][1]
        if hasattr(final_step, 'predict_proba'):
            return final_step.predict_proba(X)
        return self.predict(X)