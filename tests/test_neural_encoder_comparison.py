"""
Neural Encoder Comparison Tests

测试内容:
1. Encoder comparison 脚本结构测试
2. RankIC duplicate labels 测试
3. RankIC 股票数不一致测试
4. 标准化测试
5. Neural encoder comparison 报告测试
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEncoderComparisonScript:
    """Encoder Comparison 脚本结构测试"""

    def test_script_exists(self):
        """测试脚本文件存在"""
        script_path = 'scripts/run_neural_encoder_comparison.py'
        assert os.path.exists(script_path), "脚本文件应存在"

    def test_script_contains_all_encoders(self):
        """测试脚本包含三种编码器"""
        with open('scripts/run_neural_encoder_comparison.py', 'r', encoding='utf-8') as f:
            content = f.read()

        assert "'mlp'" in content, "应包含 MLP"
        assert "'cnn'" in content, "应包含 CNN"
        assert "'transformer'" in content, "应包含 TRANSFORMER"

    def test_script_outputs_metrics(self):
        """测试脚本输出必要指标"""
        with open('scripts/run_neural_encoder_comparison.py', 'r', encoding='utf-8') as f:
            content = f.read()

        assert 'train_loss' in content.lower(), "应输出 train_loss"
        assert 'val_loss' in content.lower(), "应输出 val_loss"
        assert 'rank_ic' in content.lower(), "应输出 RankIC"
        assert 'icir' in content.lower(), "应输出 ICIR"
        assert 'coverage' in content.lower(), "应输出 Coverage"


class TestRankICDuplicateLabels:
    """RankIC duplicate labels 测试"""

    def test_rankic_with_duplicate_labels(self):
        """测试处理重复索引"""
        from src.factors.auto.factor_evaluator import FactorEvaluator

        dates = pd.date_range('2024-01-01', periods=10, freq='B')
        stocks = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

        factor_data = []
        return_data = []

        for date in dates:
            for stock in stocks:
                factor_data.append({'date': date, 'stock': stock, 'factor': np.random.randn()})
                return_data.append({'date': date, 'stock': stock, 'return': np.random.randn() * 0.01})
            # 添加重复数据
            return_data.append({'date': date, 'stock': 'A', 'return': np.random.randn() * 0.01})
            return_data.append({'date': date, 'stock': 'B', 'return': np.random.randn() * 0.01})

        factor_df = pd.DataFrame(factor_data)
        return_df = pd.DataFrame(return_data)

        factor_series = factor_df.set_index(['date', 'stock'])['factor']
        return_series = return_df.set_index(['date', 'stock'])['return']

        evaluator = FactorEvaluator()
        result = evaluator.calculate_rank_ic(factor_series, return_series)

        assert result['count'] > 0, "RankIC count 必须 > 0"
        assert np.isfinite(result['mean']), "RankIC mean 必须有限"

    def test_rankic_no_crash_on_duplicates(self):
        """测试不会因重复标签崩溃"""
        from src.factors.auto.factor_evaluator import FactorEvaluator

        dates = pd.date_range('2024-01-01', periods=5, freq='B')
        stocks = ['A', 'B', 'C']

        factor_data = []
        return_data = []

        for date in dates:
            for stock in stocks:
                factor_data.append({'date': date, 'stock': stock, 'factor': np.random.randn()})
                return_data.append({'date': date, 'stock': stock, 'return': np.random.randn()})
            # 大量重复
            for _ in range(3):
                return_data.append({'date': date, 'stock': 'A', 'return': np.random.randn()})

        factor_series = pd.DataFrame(factor_data).set_index(['date', 'stock'])['factor']
        return_series = pd.DataFrame(return_data).set_index(['date', 'stock'])['return']

        evaluator = FactorEvaluator()

        try:
            result = evaluator.calculate_rank_ic(factor_series, return_series)
            assert True, "不应崩溃"
        except ValueError as e:
            if "cannot reindex on an axis with duplicate labels" in str(e):
                pytest.fail(f"RankIC 不应因重复标签崩溃: {e}")
            raise


class TestRankICStockMismatch:
    """RankIC 股票数不一致测试"""

    def test_rankic_with_different_stock_counts(self):
        """测试因子和收益股票数不同"""
        from src.factors.auto.factor_evaluator import FactorEvaluator

        dates = pd.date_range('2024-01-01', periods=10, freq='B')
        factor_stocks = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
        return_stocks = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']

        factor_data = []
        return_data = []

        for date in dates:
            for stock in factor_stocks:
                factor_data.append({'date': date, 'stock': stock, 'factor': np.random.randn()})
            for stock in return_stocks:
                return_data.append({'date': date, 'stock': stock, 'return': np.random.randn() * 0.01})

        factor_series = pd.DataFrame(factor_data).set_index(['date', 'stock'])['factor']
        return_series = pd.DataFrame(return_data).set_index(['date', 'stock'])['return']

        evaluator = FactorEvaluator()
        result = evaluator.calculate_rank_ic(factor_series, return_series)

        assert result['count'] > 0, "RankIC count 必须 > 0"
        assert result['count'] == len(dates), "每天都应有有效IC"

    def test_rankic_common_index_alignment(self):
        """测试按 common index 对齐"""
        from src.factors.auto.factor_evaluator import FactorEvaluator

        dates = pd.date_range('2024-01-01', periods=5, freq='B')

        common_stocks = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        factor_only_stocks = ['K', 'L', 'M', 'N']
        return_only_stocks = ['O', 'P', 'Q', 'R']

        factor_stocks = common_stocks + factor_only_stocks
        return_stocks = common_stocks + return_only_stocks

        factor_data = []
        return_data = []

        for date in dates:
            for stock in factor_stocks:
                factor_data.append({'date': date, 'stock': stock, 'factor': np.random.randn()})
            for stock in return_stocks:
                return_data.append({'date': date, 'stock': stock, 'return': np.random.randn() * 0.01})

        factor_series = pd.DataFrame(factor_data).set_index(['date', 'stock'])['factor']
        return_series = pd.DataFrame(return_data).set_index(['date', 'stock'])['return']

        evaluator = FactorEvaluator()
        result = evaluator.calculate_rank_ic(factor_series, return_series)

        assert result['count'] > 0, "应有有效IC"


class TestNormalization:
    """标准化测试"""

    def test_normalization_no_inf(self):
        """测试标准化后没有 inf"""
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
                    'volume': np.random.rand() * 1000000
                })

        df = pd.DataFrame(data)

        dataset = SequenceDataset(
            df=df,
            lookback_window=5,
            features=['open', 'high', 'low', 'close', 'volume'],
            target_horizon=1,
            normalize=True
        )

        X, metadata = dataset.get_samples()

        assert not np.any(np.isinf(X)), "标准化后不应有 inf"
        assert not np.any(np.isnan(X)), "标准化后不应有 NaN"

    def test_normalization_reasonable_loss(self):
        """测试标准化后损失数量级合理"""
        from src.factors.neural.sequence_dataset import SequenceDataset
        from src.factors.neural.autoencoder import SequenceAutoEncoder

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

        import torch
        import torch.nn as nn

        model = SequenceAutoEncoder(input_dim=2, hidden_dim=16, embedding_dim=4, lookback_window=5)
        criterion = nn.MSELoss()

        X_tensor = torch.FloatTensor(X)
        reconstruction, embedding = model(X_tensor)
        loss = criterion(reconstruction, X_tensor)

        assert loss.item() < 100, "MSE loss 不应过大"
        assert loss.item() > 0, "MSE loss 应大于0"


class TestReportRequirements:
    """报告测试"""

    def test_report_exists(self):
        """测试报告文件存在"""
        report_path = 'reports/neural_encoder_comparison.md'
        assert os.path.exists(report_path), "报告文件应存在"

    def test_report_contains_all_encoders(self):
        """测试报告包含三种编码器"""
        with open('reports/neural_encoder_comparison.md', 'r', encoding='utf-8') as f:
            content = f.read()

        assert 'MLP' in content, "报告应包含 MLP"
        assert 'CNN' in content, "报告应包含 CNN"
        assert 'TRANSFORMER' in content, "报告应包含 TRANSFORMER"

    def test_report_has_leakage_check(self):
        """测试报告包含 Leakage Check"""
        with open('reports/neural_encoder_comparison.md', 'r', encoding='utf-8') as f:
            content = f.read()

        assert 'Leakage' in content, "报告应包含 Leakage Check"
        assert 'OK' in content or 'PASS' in content, "Leakage Check 应通过"

    def test_report_has_important_notes(self):
        """测试报告包含重要提示"""
        with open('reports/neural_encoder_comparison.md', 'r', encoding='utf-8') as f:
            content = f.read()

        assert 'Important Notes' in content, "报告应包含 Important Notes"
        assert '不能直接实盘' in content, "报告应明确说明不能实盘"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])