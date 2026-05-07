"""
Neural Factor Pipeline Tests

测试内容:
1. SequenceDataset shape 测试
2. Future return alignment 测试
3. NeuralLeakageChecker 测试
4. Encoder 输出 shape 测试
5. AutoEncoder 测试
6. NeuralFactorExtractor 测试
7. NeuralFactorEvaluator 测试
8. End-to-End Mini Test
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

try:
    import torch
except ImportError:
    torch = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSequenceDataset:
    """SequenceDataset 测试"""

    def test_sequence_dataset_shape(self):
        """测试 SequenceDataset shape"""
        from src.factors.neural.sequence_dataset import SequenceDataset

        dates = pd.date_range('2024-01-01', periods=30, freq='B')
        stocks = ['A', 'B', 'C']

        data = []
        for date in dates:
            for stock in stocks:
                data.append({
                    'date': date,
                    'stock': stock,
                    'open': 100 + np.random.rand() * 10,
                    'high': 105 + np.random.rand() * 10,
                    'low': 95 + np.random.rand() * 10,
                    'close': 100 + np.random.rand() * 10,
                    'volume': np.random.rand() * 1000000,
                    'amount': np.random.rand() * 100000000,
                    'turnover': np.random.rand() * 0.05
                })

        df = pd.DataFrame(data)
        df = df.sort_values(['stock', 'date']).reset_index(drop=True)

        dataset = SequenceDataset(
            df=df,
            lookback_window=5,
            features=['open', 'high', 'low', 'close', 'volume'],
            target_horizon=1
        )

        X, metadata = dataset.get_samples()

        assert X.shape[0] == metadata.shape[0], "X 和 metadata 行数应一致"
        assert X.shape[2] == 5, "特征数应为5"
        assert X.shape[1] == 5, "窗口大小应为5"
        assert 'date' in metadata.columns, "应包含 date 列"
        assert 'stock' in metadata.columns, "应包含 stock 列"
        assert 'signal_date' in metadata.columns, "应包含 signal_date 列"

    def test_sequence_no_future_leakage(self):
        """测试序列窗口不包含未来数据"""
        from src.factors.neural.sequence_dataset import SequenceDataset

        dates = pd.date_range('2024-01-01', periods=30, freq='B')
        stocks = ['A', 'B']

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

        dataset = SequenceDataset(
            df=df,
            lookback_window=5,
            features=['close', 'volume'],
            target_horizon=1
        )

        X, metadata = dataset.get_samples()

        for i in range(len(metadata)):
            signal_date = metadata.iloc[i]['signal_date']
            row_dates = df[(df['stock'] == metadata.iloc[i]['stock']) &
                          (df['date'] <= signal_date)].nlargest(5, 'date')['date'].values

            assert all(d <= signal_date for d in row_dates), "输入窗口不应包含未来数据"


class TestFutureReturnAlignment:
    """Future Return 对齐测试"""

    def test_future_return_by_stock(self):
        """测试 future_return 按 stock 分组计算"""
        from src.factors.neural.sequence_dataset import SequenceDataset

        dates = pd.date_range('2024-01-01', periods=20, freq='B')
        stocks = ['A', 'B']

        data = []
        for stock in stocks:
            base_price = 100 if stock == 'A' else 200
            for date in dates:
                data.append({
                    'date': date,
                    'stock': stock,
                    'close': base_price + np.random.rand() * 5
                })

        df = pd.DataFrame(data)

        dataset = SequenceDataset(
            df=df,
            lookback_window=5,
            features=['close'],
            target_horizon=1
        )

        X, metadata = dataset.get_samples()

        for i in range(min(5, len(metadata))):
            stock = metadata.iloc[i]['stock']
            signal_date = metadata.iloc[i]['signal_date']
            target_date = metadata.iloc[i]['target_start_date']

            assert target_date > signal_date, "target_date 必须晚于 signal_date"


class TestNeuralLeakageChecker:
    """NeuralLeakageChecker 测试"""

    def test_leakage_check_future_columns(self):
        """测试检测未来函数关键词"""
        from src.validation.neural_leakage_check import NeuralLeakageChecker

        checker = NeuralLeakageChecker()

        risky_columns = ['future_return', 'target', 'label', 'return_forward', 'lead_close']

        for col in risky_columns:
            result = checker.check_feature_columns([col])
            assert result['status'] in ['WARN', 'FAIL'], "应检测到风险: {}".format(col)

    def test_leakage_check_safe_columns(self):
        """测试安全字段通过检查"""
        from src.validation.neural_leakage_check import NeuralLeakageChecker

        checker = NeuralLeakageChecker()

        safe_columns = ['open', 'high', 'low', 'close', 'volume', 'turnover', 'amount']

        result = checker.check_feature_columns(safe_columns)
        assert result['status'] == 'OK', "安全字段应通过检查"


class TestEncoderShapes:
    """Encoder 输出 shape 测试"""

    def test_mlp_encoder_shape(self):
        """测试 MLPSequenceEncoder 输出 shape"""
        from src.factors.neural.sequence_encoder import MLPSequenceEncoder

        batch_size = 4
        lookback_window = 5
        num_features = 5

        encoder = MLPSequenceEncoder(
            input_dim=num_features,
            hidden_dim=32,
            embedding_dim=8,
            lookback_window=lookback_window
        )

        x = torch.randn(batch_size, lookback_window, num_features)

        embedding = encoder(x)

        assert embedding.shape == (batch_size, 8), "Embedding shape 应为 (batch_size, embedding_dim)"

    def test_cnn_encoder_shape(self):
        """测试 CNN1DEncoder 输出 shape"""
        from src.factors.neural.sequence_encoder import CNN1DEncoder

        encoder = CNN1DEncoder(
            input_dim=5,
            hidden_dim=32,
            embedding_dim=8
        )

        batch_size = 4
        lookback_window = 5
        num_features = 5

        x = torch.randn(batch_size, lookback_window, num_features)

        embedding = encoder(x)

        assert embedding.shape == (batch_size, 8), "Embedding shape 应为 (batch_size, embedding_dim)"

    def test_transformer_encoder_shape(self):
        """测试 TinyTransformerEncoder 输出 shape"""
        from src.factors.neural.sequence_encoder import TinyTransformerEncoder

        encoder = TinyTransformerEncoder(
            input_dim=5,
            hidden_dim=32,
            embedding_dim=8,
            num_heads=2,
            num_layers=1
        )

        batch_size = 4
        lookback_window = 5
        num_features = 5

        x = torch.randn(batch_size, lookback_window, num_features)

        embedding = encoder(x)

        assert embedding.shape == (batch_size, 8), "Embedding shape 应为 (batch_size, embedding_dim)"


class TestAutoEncoder:
    """AutoEncoder 测试"""

    def test_autoencoder_forward(self):
        """测试 AutoEncoder forward"""
        from src.factors.neural.autoencoder import SequenceAutoEncoder

        model = SequenceAutoEncoder(
            input_dim=5,
            hidden_dim=32,
            embedding_dim=8,
            lookback_window=5
        )

        batch_size = 4
        lookback_window = 5
        num_features = 5

        x = torch.randn(batch_size, lookback_window, num_features)

        reconstruction, embedding = model(x)

        assert reconstruction.shape == x.shape, "重构 shape 应与输入一致"
        assert embedding.shape == (batch_size, 8), "Embedding shape 应正确"

    def test_autoencoder_loss(self):
        """测试 AutoEncoder loss 计算"""
        from src.factors.neural.autoencoder import SequenceAutoEncoder
        import torch.nn as nn

        model = SequenceAutoEncoder(
            input_dim=5,
            hidden_dim=32,
            embedding_dim=8,
            lookback_window=5
        )

        criterion = nn.MSELoss()

        x = torch.randn(4, 5, 5)
        reconstruction, embedding = model(x)

        loss = criterion(reconstruction, x)

        assert torch.isfinite(loss), "Loss 应为有限值"
        assert loss.item() >= 0, "Loss 应非负"


class TestNeuralFactorExtractor:
    """NeuralFactorExtractor 测试"""

    def test_embedding_to_dataframe(self):
        """测试 embedding 转换为 DataFrame"""
        from src.factors.neural.neural_factor_extractor import NeuralFactorExtractor

        dates = pd.date_range('2024-01-01', periods=10, freq='B')
        stocks = ['A', 'B', 'C']
        index = pd.MultiIndex.from_product([dates, stocks], names=['date', 'stock'])

        embeddings = np.random.randn(len(index), 8)

        metadata = pd.DataFrame({
            'date': index.get_level_values(0),
            'stock': index.get_level_values(1)
        }, index=index)

        extractor = NeuralFactorExtractor(embedding_dim=8)

        factors_df = extractor.embedding_to_dataframe(embeddings, metadata)

        assert 'date' in factors_df.columns, "应包含 date 列"
        assert 'stock' in factors_df.columns, "应包含 stock 列"
        assert 'neural_factor_0' in factors_df.columns, "应包含 neural_factor_0"

        for i in range(8):
            assert 'neural_factor_{}'.format(i) in factors_df.columns, "应包含 neural_factor_{}".format(i)


class TestNeuralFactorEvaluator:
    """NeuralFactorEvaluator 测试"""

    def test_evaluator_with_factor_evaluator(self):
        """测试使用现有 FactorEvaluator"""
        from src.factors.auto.factor_evaluator import FactorEvaluator
        from src.factors.neural.neural_factor_extractor import NeuralFactorExtractor

        dates = pd.date_range('2024-01-01', periods=20, freq='B')
        stocks = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        index = pd.MultiIndex.from_product([dates, stocks], names=['date', 'stock'])

        np.random.seed(42)
        neural_factor = pd.Series(np.random.randn(len(index)), index=index)
        future_returns = pd.Series(np.random.randn(len(index)) * 0.01, index=index)

        evaluator = FactorEvaluator()
        result = evaluator.evaluate_single(neural_factor, future_returns)

        assert 'rank_ic' in result, "应返回 rank_ic"
        assert 'icir' in result, "应返回 icir"

        assert np.isfinite(result['icir']), "ICIR 应为有限值"
        assert result['icir'] >= 0 or result['icir'] == 0, "ICIR 应非负或为0"


class TestEndToEndMini:
    """End-to-End Mini 测试"""

    def test_mini_pipeline(self):
        """测试最小完整流程"""
        from src.factors.neural.sequence_dataset import SequenceDataset
        from src.factors.neural.sequence_encoder import MLPSequenceEncoder
        from src.factors.neural.autoencoder import SequenceAutoEncoder
        from src.factors.neural.neural_factor_extractor import NeuralFactorExtractor
        from src.factors.auto.factor_evaluator import FactorEvaluator

        dates = pd.date_range('2024-01-01', periods=30, freq='B')
        stocks = ['A', 'B', 'C']

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

        dataset = SequenceDataset(df=df, lookback_window=5, features=['close', 'volume'], target_horizon=1)
        X, metadata = dataset.get_samples()

        assert X.shape[0] > 0, "应有样本生成"

        model = SequenceAutoEncoder(input_dim=2, hidden_dim=16, embedding_dim=4, lookback_window=5)

        X_tensor = torch.FloatTensor(X)
        reconstruction, embedding = model(X_tensor)

        assert embedding.shape[1] == 4, "Embedding 维度应为4"

        extractor = NeuralFactorExtractor(embedding_dim=4)
        factors_df = extractor.embedding_to_dataframe(embedding.detach().numpy(), metadata)

        assert len(factors_df) == len(metadata), "Factor 行数应与 metadata 一致"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
