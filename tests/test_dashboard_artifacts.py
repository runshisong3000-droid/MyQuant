"""
测试 Dashboard Artifacts
"""

import unittest
import os
import sys
import json
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dashboard.data_loader import DataLoader


class TestArtifactFormats(unittest.TestCase):
    """测试 artifact 格式正确性"""

    def setUp(self):
        self.loader = DataLoader()
        self.dashboard_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'dashboard')

    def test_equity_curve_format(self):
        """测试 equity_curve.parquet 格式"""
        df, error = self.loader.load_equity_curve()
        if df is not None:
            required_columns = ['date', 'portfolio_value', 'daily_return']
            for col in required_columns:
                self.assertIn(col, df.columns)
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(df['date']))

    def test_drawdown_curve_format(self):
        """测试 drawdown_curve.parquet 格式"""
        df, error = self.loader.load_drawdown_curve()
        if df is not None:
            required_columns = ['date', 'drawdown']
            for col in required_columns:
                self.assertIn(col, df.columns)

    def test_backtest_summary_format(self):
        """测试 backtest_summary.json 格式"""
        summary, error = self.loader.load_backtest_summary()
        if summary is not None:
            self.assertIn('can_use_for_live_trading', summary)
            self.assertFalse(summary['can_use_for_live_trading'])
            required_fields = ['total_return', 'annual_return', 'sharpe', 'max_drawdown']
            for field in required_fields:
                self.assertIn(field, summary)

    def test_factor_summary_format(self):
        """测试 factor_summary.parquet 格式"""
        df, error = self.loader.load_factor_summary()
        if df is not None:
            required_columns = [
                'factor_name', 'factor_type', 'rank_ic_mean',
                'icir', 'coverage', 'turnover', 'gatekeeper_status'
            ]
            for col in required_columns:
                self.assertIn(col, df.columns)

    def test_factor_ic_series_format(self):
        """测试 factor_ic_series.parquet 格式"""
        df, error = self.loader.load_factor_ic_series()
        if df is not None:
            required_columns = ['date', 'factor_name', 'rank_ic']
            for col in required_columns:
                self.assertIn(col, df.columns)

    def test_factor_correlation_format(self):
        """测试 factor_correlation.parquet 格式"""
        df, error = self.loader.load_factor_correlation()
        if df is not None:
            required_columns = ['factor_1', 'factor_2', 'correlation']
            for col in required_columns:
                self.assertIn(col, df.columns)

    def test_neural_factors_format(self):
        """测试 neural_factors.parquet 格式"""
        df, error = self.loader.load_neural_factors_dashboard()
        if df is not None:
            required_columns = ['date', 'stock']
            for col in required_columns:
                self.assertIn(col, df.columns)
            factor_cols = [col for col in df.columns if 'neural_factor_' in col]
            self.assertTrue(len(factor_cols) > 0)

    def test_neural_factor_summary_format(self):
        """测试 neural_factor_summary.parquet 格式"""
        df, error = self.loader.load_neural_factor_summary()
        if df is not None:
            required_columns = ['factor_name', 'encoder_type', 'rank_ic_mean', 'icir', 'coverage']
            for col in required_columns:
                self.assertIn(col, df.columns)

    def test_encoder_comparison_format(self):
        """测试 encoder_comparison.parquet 格式"""
        df, error = self.loader.load_encoder_comparison_data()
        if df is not None:
            required_columns = [
                'encoder', 'train_loss', 'val_loss', 'avg_rankic',
                'best_rankic', 'avg_icir', 'best_icir', 'passing_factors'
            ]
            for col in required_columns:
                self.assertIn(col, df.columns)


class TestArtifactLoadSafety(unittest.TestCase):
    """测试 artifact 加载安全性"""

    def setUp(self):
        self.loader = DataLoader()

    def test_equity_curve_can_load(self):
        """测试 equity_curve 可以加载"""
        df, error = self.loader.load_equity_curve()
        if error is None:
            self.assertIsNotNone(df)

    def test_drawdown_curve_can_load(self):
        """测试 drawdown_curve 可以加载"""
        df, error = self.loader.load_drawdown_curve()
        if error is None:
            self.assertIsNotNone(df)

    def test_factor_ic_series_can_load(self):
        """测试 factor_ic_series 可以加载"""
        df, error = self.loader.load_factor_ic_series()
        if error is None:
            self.assertIsNotNone(df)

    def test_factor_summary_can_load(self):
        """测试 factor_summary 可以加载"""
        df, error = self.loader.load_factor_summary()
        if error is None:
            self.assertIsNotNone(df)

    def test_neural_factors_can_load(self):
        """测试 neural_factors 可以加载"""
        df, error = self.loader.load_neural_factors_dashboard()
        if error is None:
            self.assertIsNotNone(df)

    def test_neural_factor_summary_can_load(self):
        """测试 neural_factor_summary 可以加载"""
        df, error = self.loader.load_neural_factor_summary()
        if error is None:
            self.assertIsNotNone(df)

    def test_encoder_comparison_can_load(self):
        """测试 encoder_comparison 可以加载"""
        df, error = self.loader.load_encoder_comparison_data()
        if error is None:
            self.assertIsNotNone(df)

    def test_backtest_summary_can_load(self):
        """测试 backtest_summary 可以加载"""
        data, error = self.loader.load_backtest_summary()
        if error is None:
            self.assertIsNotNone(data)


class TestManifestStatus(unittest.TestCase):
    """测试 manifest 状态正确性"""

    def setUp(self):
        self.loader = DataLoader()

    def test_manifest_reflects_reality(self):
        """测试 manifest 能正确反映文件存在状态"""
        manifest, _ = self.loader.load_dashboard_manifest()
        if manifest and 'artifacts' in manifest:
            for filename, info in manifest['artifacts'].items():
                filepath = os.path.join(os.path.dirname(__file__), '..', 'data', 'dashboard', filename)
                exists_on_disk = os.path.exists(filepath)
                self.assertEqual(info.get('exists', False), exists_on_disk,
                                f"Manifest mismatch for {filename}")

    def test_cannot_use_for_live_trading(self):
        """测试 can_use_for_live_trading 为 false"""
        manifest, _ = self.loader.load_dashboard_manifest()
        if manifest:
            self.assertFalse(manifest.get('can_use_for_live_trading', True))


class TestNoFakeData(unittest.TestCase):
    """测试不伪造数据"""

    def setUp(self):
        self.loader = DataLoader()

    def test_backtest_summary_real(self):
        """测试 backtest_summary 是真实的"""
        summary, error = self.loader.load_backtest_summary()
        if summary is not None:
            self.assertIn('total_return', summary)
            self.assertIn('sharpe', summary)
            self.assertIn('max_drawdown', summary)


if __name__ == '__main__':
    unittest.main()
