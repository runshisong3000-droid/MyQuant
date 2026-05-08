"""
测试 Phase 2.3 Artifact 验证
"""

import unittest
import os
import sys
import json
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dashboard.data_loader import DataLoader


class TestPhase23ArtifactValidation(unittest.TestCase):
    """测试 Phase 2.3 artifacts"""
    
    def setUp(self):
        self.loader = DataLoader()
        self.dashboard_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'dashboard')
    
    def test_dashboard_manifest_format(self):
        """测试 dashboard_manifest.json 格式正确"""
        manifest, error = self.loader.load_dashboard_manifest()
        if manifest is not None:
            self.assertIn('version', manifest)
            self.assertIn('generated_at', manifest)
            self.assertIn('artifacts', manifest)
            self.assertIn('can_use_for_live_trading', manifest)
            self.assertFalse(manifest['can_use_for_live_trading'])
    
    def test_backtest_summary_cannot_trade(self):
        """测试 backtest_summary.json can_use_for_live_trading 必须是 false"""
        summary, error = self.loader.load_backtest_summary()
        if summary is not None:
            self.assertIn('can_use_for_live_trading', summary)
            self.assertFalse(summary['can_use_for_live_trading'])
    
    def test_equity_curve_columns(self):
        """测试 equity_curve.parquet 必须包含 date 和 portfolio_value"""
        df, error = self.loader.load_equity_curve()
        if df is not None:
            self.assertIn('date', df.columns)
            self.assertIn('portfolio_value', df.columns)
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(df['date']))
    
    def test_drawdown_curve_columns(self):
        """测试 drawdown_curve.parquet 必须包含 date 和 drawdown"""
        df, error = self.loader.load_drawdown_curve()
        if df is not None:
            self.assertIn('date', df.columns)
            self.assertIn('drawdown', df.columns)
    
    def test_factor_summary_columns(self):
        """测试 factor_summary.parquet 必须包含 factor_name、rank_ic_mean、icir"""
        df, error = self.loader.load_factor_summary()
        if df is not None:
            self.assertIn('factor_name', df.columns)
            self.assertIn('rank_ic_mean', df.columns)
            self.assertIn('icir', df.columns)
    
    def test_factor_ic_series_columns(self):
        """测试 factor_ic_series.parquet 必须包含 date、factor_name、rank_ic"""
        df, error = self.loader.load_factor_ic_series()
        if df is not None:
            self.assertIn('date', df.columns)
            self.assertIn('factor_name', df.columns)
            self.assertIn('rank_ic', df.columns)
    
    def test_factor_correlation_columns(self):
        """测试 factor_correlation.parquet 必须包含 factor_1、factor_2、correlation"""
        df, error = self.loader.load_factor_correlation()
        if df is not None:
            self.assertIn('factor_1', df.columns)
            self.assertIn('factor_2', df.columns)
            self.assertIn('correlation', df.columns)
    
    def test_neural_factors_columns(self):
        """测试 neural_factors.parquet 必须包含 date、stock"""
        df, error = self.loader.load_neural_factors_dashboard()
        if df is not None:
            self.assertIn('date', df.columns)
            self.assertIn('stock', df.columns)
    
    def test_neural_factor_summary_columns(self):
        """测试 neural_factor_summary.parquet 必须包含 factor_name、rank_ic_mean、icir"""
        df, error = self.loader.load_neural_factor_summary()
        if df is not None:
            self.assertIn('factor_name', df.columns)
            self.assertIn('rank_ic_mean', df.columns)
            self.assertIn('icir', df.columns)
    
    def test_encoder_comparison_columns(self):
        """测试 encoder_comparison.parquet 必须包含 encoder、train_loss、val_loss、avg_rankic"""
        df, error = self.loader.load_encoder_comparison_data()
        if df is not None:
            self.assertIn('encoder', df.columns)
            self.assertIn('train_loss', df.columns)
            self.assertIn('val_loss', df.columns)
            self.assertIn('avg_rankic', df.columns)
    
    def test_missing_artifact_not_crash(self):
        """测试缺失 artifact 不会导致崩溃"""
        methods = [
            self.loader.load_equity_curve,
            self.loader.load_drawdown_curve,
            self.loader.load_backtest_summary,
            self.loader.load_factor_summary,
            self.loader.load_factor_ic_series,
            self.loader.load_factor_correlation,
            self.loader.load_neural_factors_dashboard,
            self.loader.load_neural_factor_summary,
            self.loader.load_encoder_comparison_data
        ]
        
        for method in methods:
            result, error = method()
            self.assertTrue(result is None or isinstance(result, (pd.DataFrame, dict)))
    
    def test_no_can_use_for_live_trading_true(self):
        """测试不允许 can_use_for_live_trading=true"""
        # Check manifest
        manifest, _ = self.loader.load_dashboard_manifest()
        if manifest:
            self.assertFalse(manifest.get('can_use_for_live_trading', True))
        
        # Check reliability status
        reliability_status, _ = self.loader.load_json('reliability_status.json', directory='dashboard')
        if reliability_status:
            self.assertFalse(reliability_status.get('can_use_for_live_trading', True))
        
        # Check backtest summary
        backtest_summary, _ = self.loader.load_backtest_summary()
        if backtest_summary:
            self.assertFalse(backtest_summary.get('can_use_for_live_trading', True))


if __name__ == '__main__':
    unittest.main()
