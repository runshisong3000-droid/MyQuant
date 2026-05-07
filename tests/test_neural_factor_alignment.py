"""
Neural Factor Alignment Test

测试 neural factor 与 future_return 的对齐问题
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestNeuralFactorAlignment:
    """测试 neural factor 与 future_return 的对齐"""

    def test_alignment_with_small_data(self):
        """使用小样本测试对齐"""
        from src.factors.neural.sequence_dataset import SequenceDataset
        from src.factors.neural.neural_factor_extractor import NeuralFactorExtractor
        from src.factors.auto.factor_evaluator import FactorEvaluator

        dates = pd.date_range('2024-01-01', periods=40, freq='B')
        stocks = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']  # 12只股票

        data = []
        for date in dates:
            for stock in stocks:
                data.append({
                    'date': date,
                    'stock': stock,
                    'close': 100 + np.random.rand() * 10,
                    'volume': np.random.rand() * 1000000
                })

        df = pd.DataFrame(data)
        df = df.sort_values(['stock', 'date']).reset_index(drop=True)

        lookback_window = 5
        horizon = 1

        dataset = SequenceDataset(
            df=df,
            lookback_window=lookback_window,
            features=['close', 'volume'],
            target_horizon=horizon
        )

        X, metadata = dataset.get_samples()

        assert len(metadata) > 0, "应有样本生成"

        meta_test = metadata

        print("\n[TEST DIAGNOSTIC]")
        print("  - metadata shape: {}".format(metadata.shape))
        print("  - signal_date range: {} to {}".format(
            metadata['signal_date'].min(), metadata['signal_date'].max()))
        print("  - stock unique: {}".format(metadata['stock'].nunique()))

        horizon = 1

        df_with_future = df.copy()
        df_with_future['future_return'] = df_with_future.groupby('stock')['close'].pct_change().shift(-horizon)

        future_returns = df_with_future[['date', 'stock', 'future_return']].dropna()

        print("  - future_returns shape: {}".format(future_returns.shape))

        factors_df = pd.DataFrame({
            'signal_date': meta_test['signal_date'].values,
            'stock': meta_test['stock'].values,
            'neural_factor_0': np.random.randn(len(meta_test))
        })

        print("  - factors_df shape: {}".format(factors_df.shape))

        factor_series = factors_df.set_index(['signal_date', 'stock'])['neural_factor_0']

        future_series = future_returns.set_index(['date', 'stock'])['future_return']

        print("  - factor_series index: {}".format(factor_series.index.names))
        print("  - future_series index: {}".format(future_series.index.names))

        common_idx = factor_series.index.intersection(future_series.index)
        print("  - Common index count: {}".format(len(common_idx)))

        assert len(common_idx) > 0, "应该有共同的index!"

        evaluator = FactorEvaluator()
        result = evaluator.evaluate_single(factor_series, future_series)

        print("  - RankIC count: {}".format(result['rank_ic']['count']))

        assert result['rank_ic']['count'] > 0, "RankIC count 应该大于 0!"

    def test_alignment_requires_multindex(self):
        """测试对齐需要 MultiIndex"""
        from src.factors.neural.sequence_dataset import SequenceDataset

        dates = pd.date_range('2024-01-01', periods=30, freq='B')
        stocks = ['A', 'B']

        data = []
        for date in dates:
            for stock in stocks:
                data.append({
                    'date': date,
                    'stock': stock,
                    'close': 100 + np.random.rand() * 10
                })

        df = pd.DataFrame(data)

        dataset = SequenceDataset(df=df, lookback_window=5, features=['close'], target_horizon=1)
        X, metadata = dataset.get_samples()

        assert 'signal_date' in metadata.columns, "metadata应有signal_date列"
        assert 'stock' in metadata.columns, "metadata应有stock列"

    def test_future_return_by_stock_groupby(self):
        """测试 future_return 必须按 stock 分组计算"""
        dates = pd.date_range('2024-01-01', periods=20, freq='B')
        stocks = ['A', 'B']

        data = []
        for stock in stocks:
            base = 100 if stock == 'A' else 200
            for date in dates:
                data.append({
                    'date': date,
                    'stock': stock,
                    'close': base + np.random.rand() * 5
                })

        df = pd.DataFrame(data)
        df = df.sort_values(['stock', 'date']).reset_index(drop=True)

        future_returns = df.groupby('stock').apply(
            lambda x: x.set_index('date')['close'].pct_change().shift(-1)
        ).reset_index(level=0)

        assert 'stock' in future_returns.columns, "应有stock列"
        assert len(future_returns) > 0, "应有future_returns数据"

    def test_evaluator_with_aligned_data(self):
        """测试 FactorEvaluator 能处理对齐的数据"""
        from src.factors.auto.factor_evaluator import FactorEvaluator

        dates = pd.date_range('2024-01-01', periods=15, freq='B')
        stocks = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']

        index = pd.MultiIndex.from_product([dates, stocks], names=['date', 'stock'])

        np.random.seed(42)
        factor = pd.Series(np.random.randn(len(index)), index=index)
        future_return = pd.Series(np.random.randn(len(index)) * 0.01, index=index)

        evaluator = FactorEvaluator()
        result = evaluator.evaluate_single(factor, future_return)

        assert result['rank_ic']['count'] > 0, "RankIC count应该大于0"
        assert np.isfinite(result['icir']), "ICIR应该是有限值"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
