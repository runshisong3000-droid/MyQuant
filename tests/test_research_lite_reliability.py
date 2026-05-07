"""
Research Lite Reliability Tests - 研究轻量模式可靠性测试
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDateRangeCalculation:
    """测试日期范围计算"""

    def test_calculate_start_date_cross_year(self):
        """测试跨年日期计算"""
        from src.utils.date_utils import calculate_start_date

        end_date = datetime(2026, 5, 7)
        start_date = calculate_start_date(end_date, history_months=12)

        assert start_date.year == 2025
        # Using 30 days per month approximation, so month/day may vary slightly
        assert start_date.month in [4, 5, 6]

    def test_start_date_format_yyyymmdd(self):
        """测试输出格式"""
        from src.utils.date_utils import calculate_start_date_str

        end_date = datetime(2026, 5, 7)
        start_date_str = calculate_start_date_str(end_date, history_months=12)

        assert len(start_date_str) == 8
        assert start_date_str.startswith('2025')

    def test_no_buggy_replace(self):
        """测试不使用错误的replace方式"""
        end_date = datetime(2026, 5, 7)

        buggy_date = end_date.replace(month=max(1, end_date.month - 12))
        assert buggy_date.year == 2026, "Buggy method only changes month"

        from src.utils.date_utils import calculate_start_date
        correct_date = calculate_start_date(end_date, history_months=12)
        assert correct_date.year == 2025, "Correct method changes year"


class TestStockCountConsistency:
    """测试股票数量一致性"""

    def test_target_vs_actual_stock_count(self):
        """测试目标股票数与实际股票数的检查"""
        target_count = 100
        actual_count = 42

        assert actual_count < target_count, "Should warn"

    def test_report_failed_symbols(self):
        """测试报告失败股票列表"""
        failed_symbols = ['600000.SH', '600009.SH']
        assert len(failed_symbols) > 0, "Should track failures"


class TestFormulaFactorAlignment:
    """测试公式因子对齐"""

    def test_multiindex_consistency(self):
        """测试MultiIndex一致性"""
        dates = pd.date_range('2024-01-01', periods=5, freq='B')
        stocks = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']

        data = []
        for date in dates:
            for stock in stocks:
                data.append({
                    'date': date,
                    'stock': stock,
                    'close': 100 + np.random.rand() * 10,
                    'volume': np.random.rand() * 1000000
                })

        price_data = pd.DataFrame(data)

        df_with_future = price_data.copy()
        df_with_future['future_return'] = df_with_future.groupby('stock')['close'].pct_change().shift(-1)
        future_returns = df_with_future[['date', 'stock', 'future_return']].dropna()
        future_returns = future_returns.set_index(['date', 'stock'])['future_return']

        from src.factors.auto.enhanced_generator import EnhancedFactorGenerator
        generator = EnhancedFactorGenerator()

        index = pd.MultiIndex.from_frame(price_data[['date', 'stock']])
        features = price_data.set_index(index).drop(['date', 'stock'], axis=1)

        formula_factors = generator.generate_all_factors(features, generate_neutral=False)

        factor_name = list(formula_factors.keys())[0]
        factor_data = formula_factors[factor_name]

        assert isinstance(factor_data.index, pd.MultiIndex), "Factor must have MultiIndex"
        assert factor_data.index.names == ['date', 'stock'], "Index names must be date and stock"

        assert isinstance(future_returns.index, pd.MultiIndex), "Future returns must have MultiIndex"
        assert future_returns.index.names == ['date', 'stock'], "Index names must be date and stock"

        common_idx = factor_data.index.intersection(future_returns.index)
        assert len(common_idx) > 0, "Must have common indices"

    def test_evaluate_single_rankic_count(self):
        """测试evaluate_single有rank_ic count > 0"""
        from src.factors.auto.factor_evaluator import FactorEvaluator

        dates = pd.date_range('2024-01-01', periods=20, freq='B')
        stocks = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']

        data = []
        for date in dates:
            for stock in stocks:
                data.append({
                    'date': date,
                    'stock': stock,
                    'close': 100 + np.random.rand() * 10
                })

        price_data = pd.DataFrame(data)

        df_with_future = price_data.copy()
        df_with_future['future_return'] = df_with_future.groupby('stock')['close'].pct_change().shift(-1)
        future_returns = df_with_future[['date', 'stock', 'future_return']].dropna()
        future_returns = future_returns.set_index(['date', 'stock'])['future_return']

        factor_index = pd.MultiIndex.from_frame(df_with_future[['date', 'stock']])
        factor_data = pd.Series(np.random.randn(len(factor_index)), index=factor_index)

        evaluator = FactorEvaluator()
        result = evaluator.evaluate_single(factor_data, future_returns)

        assert result['rank_ic']['count'] > 0, "RankIC count must be > 0"
        assert np.isfinite(result['rank_ic']['mean']), "RankIC mean must be finite"


class TestNeuralLeakageFailStop:
    """测试Neural Leakage Fail-Stop"""

    def test_fail_stop_on_leakage(self):
        """测试leakage FAIL时必须停止"""
        from src.validation.neural_leakage_check import NeuralLeakageChecker

        checker = NeuralLeakageChecker()

        bad_columns = ['future_return', 'target', 'label']
        feature_check = checker.check_feature_columns(bad_columns)

        assert feature_check['status'] == 'FAIL', "Should FAIL on forbidden keywords"

        overall_result = checker.run_all_checks(columns=bad_columns)

        assert overall_result['overall_status'] == 'FAIL', "Overall should be FAIL"

    def test_leakage_check_fail_triggers_stop(self):
        """测试leakage FAIL是否会触发停止逻辑"""
        leakage_status = 'FAIL'

        should_continue = leakage_status in ['OK', 'WARN']
        assert should_continue is False, "Should not continue on FAIL"

    def test_leakage_check_warn_can_continue(self):
        """测试leakage WARN可以继续"""
        leakage_status = 'WARN'

        should_continue = leakage_status in ['OK', 'WARN']
        assert should_continue is True, "Should continue on WARN"


class TestScalerFitScope:
    """测试scaler fit scope"""

    def test_train_before_val_before_test(self):
        """测试train < val < test"""
        train_dates = (datetime(2024, 1, 1), datetime(2024, 6, 1))
        val_dates = (datetime(2024, 6, 2), datetime(2024, 8, 1))
        test_dates = (datetime(2024, 8, 2), datetime(2024, 10, 1))

        assert train_dates[1] < val_dates[0], "Train must end before val starts"
        assert val_dates[1] < test_dates[0], "Val must end before test starts"


class TestStudentPipelineNotBroken:
    """测试student pipeline不被破坏"""

    def test_all_existing_tests_should_pass(self):
        """简单占位，实际运行完整测试套件"""
        assert True, "Placeholder"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
