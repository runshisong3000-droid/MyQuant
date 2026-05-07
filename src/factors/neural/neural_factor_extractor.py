"""
Neural Factor Extractor - 神经因子提取器

将embedding转换为DataFrame格式的候选因子。
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
import json
from datetime import datetime


class NeuralFactorExtractor:
    """
    神经因子提取器

    将encoder输出的embedding转换为DataFrame格式的候选因子。

    Attributes:
        embedding_dim: embedding维度，决定生成的因子数量
    """

    def __init__(self, embedding_dim: int = 8):
        self.embedding_dim = embedding_dim

    def embedding_to_dataframe(
        self,
        embeddings: np.ndarray,
        metadata: pd.DataFrame
    ) -> pd.DataFrame:
        """
        将embedding转换为DataFrame

        Args:
            embeddings: shape (num_samples, embedding_dim)
            metadata: DataFrame with 'date' and 'stock' columns

        Returns:
            DataFrame with date, stock, neural_factor_0, ..., neural_factor_{embedding_dim-1}
        """
        if len(embeddings) != len(metadata):
            raise ValueError(
                "Embeddings ({}) and metadata ({}) length mismatch".format(
                    len(embeddings), len(metadata)
                )
            )

        result = metadata.copy()

        for i in range(self.embedding_dim):
            result['neural_factor_{}'.format(i)] = embeddings[:, i]

        return result

    def save_factors(
        self,
        factors_df: pd.DataFrame,
        output_path: str = 'data/factors/neural_factors.parquet'
    ) -> None:
        """
        保存因子到parquet文件

        Args:
            factors_df: 因子DataFrame
            output_path: 输出路径
        """
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        factors_df.to_parquet(output_path, index=False)
        print("Neural factors saved to: {}".format(output_path))

    def save_metadata(
        self,
        metadata_dict: Dict[str, Any],
        output_path: str = 'reports/neural_factor_metadata.json'
    ) -> None:
        """
        保存元数据到JSON文件

        Args:
            metadata_dict: 元数据字典
            output_path: 输出路径
        """
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, indent=2, default=str)

        print("Neural factor metadata saved to: {}".format(output_path))

    def create_metadata(
        self,
        model_type: str,
        lookback_window: int,
        raw_features: list,
        embedding_dim: int,
        train_start_date: Any,
        train_end_date: Any,
        validation_start_date: Any,
        validation_end_date: Any,
        test_start_date: Any,
        test_end_date: Any,
        horizon: int,
        training_mode: str,
        device: str,
        leakage_check_result: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        创建元数据字典

        Returns:
            元数据字典
        """
        return {
            'model_type': model_type,
            'lookback_window': lookback_window,
            'raw_features': raw_features,
            'embedding_dim': embedding_dim,
            'train_start_date': str(train_start_date),
            'train_end_date': str(train_end_date),
            'validation_start_date': str(validation_start_date),
            'validation_end_date': str(validation_end_date),
            'test_start_date': str(test_start_date),
            'test_end_date': str(test_end_date),
            'horizon': horizon,
            'training_mode': training_mode,
            'device': device,
            'leakage_check_result': leakage_check_result,
            'generated_at': datetime.now().isoformat()
        }
