"""
可靠性修复测试

测试内容:
1. ICIR 正常性测试
2. 未来函数检查测试
3. MultiIndex 回测测试
4. Sharpe 边界测试
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime


class TestICIRReliability:
    """ICIR 计算正常性测试"""

    def test_icir_basic_calculation(self):
        """测试基本 ICIR 计算"""
        dates = pd.date_range('2024-01-01', periods=3, freq='B')
        stocks = ['A', 'B', 'C', 'D', 'E']
        
        index = pd.MultiIndex.from_product([dates, stocks], names=['date', 'stock'])
        
        np.random.seed(42)
        factor = pd.Series(np.random.randn(len(index)), index=index)
        future_return = pd.Series(np.random.randn(len(index)) * 0.01, index=index)
        
        from src.factors.auto.factor_evaluator import FactorEvaluator
        evaluator = FactorEvaluator()
        
        result = evaluator.evaluate_single(factor, future_return)
        
        icir = result.get('icir', 0)
        
        assert not np.isinf(icir), f"ICIR should not be inf, got {icir}"
        assert np.isfinite(icir), f"ICIR should be finite, got {icir}"
        assert np.abs(icir) < 1000, f"ICIR should be reasonable, got {icir}"

    def test_icir_with_zero_std(self):
        """测试 IC 标准差为 0 时的处理"""
        dates = pd.date_range('2024-01-01', periods=3, freq='B')
        stocks = ['A', 'B', 'C', 'D', 'E']
        
        index = pd.MultiIndex.from_product([dates, stocks], names=['date', 'stock'])
        
        factor = pd.Series(np.ones(len(index)), index=index)
        future_return = pd.Series(np.random.randn(len(index)) * 0.01, index=index)
        
        from src.factors.auto.factor_evaluator import FactorEvaluator
        evaluator = FactorEvaluator()
        
        result = evaluator.evaluate_single(factor, future_return)
        
        icir = result.get('icir', 0)
        
        assert icir == 0 or np.isnan(icir), f"ICIR with constant factor should be 0 or NaN, got {icir}"

    def test_icir_empty_ic_values(self):
        """测试 IC 值为空时的处理"""
        factor = pd.Series([], dtype=float)
        factor.index = pd.MultiIndex.from_tuples([], names=['date', 'stock'])
        future_return = factor.copy()
        
        from src.factors.auto.factor_evaluator import FactorEvaluator
        evaluator = FactorEvaluator()
        
        result = evaluator.evaluate_single(factor, future_return)
        
        icir = result.get('icir', 0)
        assert icir == 0, f"ICIR with empty data should be 0, got {icir}"


class TestFutureLeakageDetection:
    """未来函数检测测试"""

    def test_leakage_detection_basic(self):
        """测试基本未来函数检测"""
        from src.factors.auto.factor_evaluator import FactorEvaluator
        
        evaluator = FactorEvaluator()
        
        # Test with suspicious formula names
        suspicious_names = [
            'future_return',
            'target_return',
            'label_1d',
            'close_shift_-1',
            'lead_5d'
        ]
        
        safe_names = [
            'momentum_20d',
            'volume_ma_10',
            'std_30d',
            'reversal_5d'
        ]
        
        for name in suspicious_names:
            has_risk = evaluator._check_future_leakage(name)
            assert has_risk, f"Should detect leakage in: {name}"
        
        for name in safe_names:
            has_risk = evaluator._check_future_leakage(name)
            assert not has_risk, f"Should not detect leakage in: {name}"


class TestMultiIndexBacktest:
    """MultiIndex 回测测试"""

    def test_multiindex_backtest_basic(self):
        """测试 MultiIndex 数据的回测处理"""
        dates = pd.date_range('2024-01-01', periods=10, freq='B')
        stocks = ['A', 'B', 'C', 'D', 'E']
        
        index = pd.MultiIndex.from_product([dates, stocks], names=['date', 'stock'])
        
        np.random.seed(42)
        data = pd.DataFrame({
            'close': 100 + np.random.randn(len(index)) * 2,
            'signal': np.random.rand(len(index)),
            'weight': 0.2
        }, index=index)
        
        daily_groups = data.groupby(level='date')
        
        assert len(list(daily_groups)) == len(dates), "Should have group for each date"
        
        for date, group in daily_groups:
            assert 'close' in group.columns
            assert 'signal' in group.columns
            assert len(group) == len(stocks), f"Group size mismatch for {date}"

    def test_multiindex_to_single_index(self):
        """测试 MultiIndex 转换为单级 index"""
        dates = pd.date_range('2024-01-01', periods=3, freq='B')
        stocks = ['A', 'B', 'C']
        
        index = pd.MultiIndex.from_product([dates, stocks], names=['date', 'stock'])
        
        series = pd.Series(np.random.rand(len(index)), index=index)
        
        # Test xs operation
        for date in dates:
            day_data = series.xs(date, level='date')
            assert len(day_data) == len(stocks), f"xs operation failed for {date}"


class TestSharpeReliability:
    """Sharpe 比率边界测试"""

    def test_sharpe_normal(self):
        """测试正常收益率的 Sharpe"""
        returns = pd.Series(np.random.randn(100) * 0.01)
        
        from src.metrics.performance import calculate_sharpe
        
        sharpe = calculate_sharpe(returns)
        
        assert np.isfinite(sharpe), f"Sharpe should be finite, got {sharpe}"
        assert np.abs(sharpe) < 100, f"Sharpe should be reasonable, got {sharpe}"

    def test_sharpe_all_zeros(self):
        """测试全零收益率"""
        returns = pd.Series(np.zeros(100))
        
        from src.metrics.performance import calculate_sharpe
        
        sharpe = calculate_sharpe(returns)
        
        assert sharpe == 0, f"Sharpe with zero returns should be 0, got {sharpe}"

    def test_sharpe_with_nan(self):
        """测试包含 NaN 的收益率"""
        returns = pd.Series([0.01, 0.02, np.nan, 0.01, 0.02])
        
        from src.metrics.performance import calculate_sharpe
        
        sharpe = calculate_sharpe(returns)
        
        assert np.isfinite(sharpe), f"Sharpe should handle NaN, got {sharpe}"

    def test_sharpe_with_inf(self):
        """测试包含 inf 的收益率"""
        returns = pd.Series([0.01, np.inf, 0.01])
        
        from src.metrics.performance import calculate_sharpe
        
        sharpe = calculate_sharpe(returns)
        
        assert np.isfinite(sharpe), f"Sharpe should handle inf, got {sharpe}"

    def test_sharpe_empty_series(self):
        """测试空序列"""
        returns = pd.Series([], dtype=float)
        
        from src.metrics.performance import calculate_sharpe
        
        sharpe = calculate_sharpe(returns)
        
        assert sharpe == 0, f"Sharpe with empty series should be 0, got {sharpe}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
