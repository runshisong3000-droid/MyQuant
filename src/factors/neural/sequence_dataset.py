"""
Sequence Dataset - 序列数据构造

将DataFrame转换为可用于神经网络训练的序列样本。
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any


class SequenceDataset:
    """
    序列数据集

    将OHLCV数据转换为 (X, metadata) 格式的序列样本。

    Attributes:
        df: 原始DataFrame
        lookback_window: 回看窗口大小
        features: 用于输入的特征列表
        target_horizon: 预测期限
    """

    def __init__(
        self,
        df: pd.DataFrame,
        lookback_window: int = 20,
        features: List[str] = None,
        target_horizon: int = 1,
        normalize: bool = True
    ):
        if features is None:
            features = ['open', 'high', 'low', 'close', 'volume']

        self.df = df.copy()
        self.lookback_window = lookback_window
        self.features = features
        self.target_horizon = target_horizon
        self.normalize = normalize
        self.feature_means = None
        self.feature_stds = None

        self.df = self.df.sort_values(['stock', 'date']).reset_index(drop=True)

        self._validate_columns()

        self.stocks = self.df['stock'].unique()
        self.dates = sorted(self.df['date'].unique())

        if self.normalize:
            self._compute_normalization_stats()

        self.samples = self._build_samples()

    def _validate_columns(self):
        """验证必需的列"""
        required_cols = ['date', 'stock'] + self.features
        missing = set(required_cols) - set(self.df.columns)
        if missing:
            raise ValueError("Missing columns: {}".format(missing))

    def _compute_normalization_stats(self):
        """计算标准化统计量 (在训练集上)"""
        all_features = self.df[self.features].values
        self.feature_means = np.nanmean(all_features, axis=0)
        self.feature_stds = np.nanstd(all_features, axis=0)
        self.feature_stds[self.feature_stds == 0] = 1.0

    def _normalize_features(self, X):
        """标准化特征"""
        if not self.normalize or self.feature_means is None:
            return X
        return (X - self.feature_means) / self.feature_stds

    def _build_samples(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        构建序列样本

        Returns:
            X: shape (num_samples, lookback_window, num_features)
            metadata: DataFrame with date, stock, signal_date, target_start_date, target_end_date
        """
        samples = []
        metadata_list = []

        lookback = self.lookback_window
        horizon = self.target_horizon

        for stock in self.stocks:
            stock_df = self.df[self.df['stock'] == stock].sort_values('date').reset_index(drop=True)

            stock_dates = stock_df['date'].values
            stock_features = stock_df[self.features].values

            n = len(stock_df)

            for i in range(lookback, n - horizon):
                start_idx = i - lookback
                end_idx = i

                X_sample = stock_features[start_idx:end_idx]
                if self.normalize:
                    X_sample = self._normalize_features(X_sample)

                signal_date = stock_dates[end_idx - 1]
                target_start_date = stock_dates[end_idx]
                target_end_date = stock_dates[end_idx + horizon - 1] if horizon > 1 else target_start_date

                if target_start_date <= signal_date:
                    continue

                samples.append(X_sample)

                metadata_list.append({
                    'date': signal_date,
                    'stock': stock,
                    'signal_date': signal_date,
                    'target_start_date': target_start_date,
                    'target_end_date': target_end_date
                })

        X = np.array(samples, dtype=np.float32)
        metadata = pd.DataFrame(metadata_list)

        return X, metadata

    def get_samples(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        获取样本

        Returns:
            X: shape (num_samples, lookback_window, num_features)
            metadata: DataFrame with date, stock, signal_date, target_start_date, target_end_date
        """
        return self.samples

    def get_train_val_test_split(
        self,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2
    ) -> Dict[str, Tuple[np.ndarray, pd.DataFrame]]:
        """
        按时间顺序切分训练集、验证集、测试集

        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例

        Returns:
            dict with 'train', 'val', 'test' keys
        """
        X, metadata = self.samples

        if len(X) == 0:
            raise ValueError("No samples available for splitting")

        unique_dates = sorted(metadata['signal_date'].unique())
        n_dates = len(unique_dates)

        if n_dates < 3:
            raise ValueError("Need at least 3 unique dates for train/val/test split")

        n_train = int(n_dates * train_ratio)
        n_val = int(n_dates * val_ratio)

        train_dates = unique_dates[:n_train]
        val_dates = unique_dates[n_train:n_train + n_val]
        test_dates = unique_dates[n_train + n_val:]

        train_mask = metadata['signal_date'].isin(train_dates)
        val_mask = metadata['signal_date'].isin(val_dates)
        test_mask = metadata['signal_date'].isin(test_dates)

        return {
            'train': (X[train_mask], metadata[train_mask]),
            'val': (X[val_mask], metadata[val_mask]),
            'test': (X[test_mask], metadata[test_mask])
        }
