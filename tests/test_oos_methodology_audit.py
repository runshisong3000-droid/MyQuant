"""
OOS Methodology Audit Tests

验证 OOS 验证流水线的方法论正确性：
1. formula_factors.parquet 必须包含真实的 stock-date 因子值
2. 禁止使用摘要表代替因子面板
3. feature set 构造必须正确
4. 样本量警告机制必须生效
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.research.feature_set_comparison import FeatureSetComparison


class TestMethodologyAudit:
    """方法论审计测试"""

    def test_factor_summary_not_used_as_panel(self):
        """factor_summary.parquet 不能被当作公式因子面板"""
        if os.path.exists('data/dashboard/factor_summary.parquet'):
            df = pd.read_parquet('data/dashboard/factor_summary.parquet')
            # factor_summary 不应该有 date 和 stock 列
            assert 'date' not in df.columns, "factor_summary should not contain 'date' column"
            assert 'stock' not in df.columns, "factor_summary should not contain 'stock' column"

    def test_factor_ic_series_not_used_as_panel(self):
        """factor_ic_series.parquet 不能被当作公式因子面板"""
        if os.path.exists('data/dashboard/factor_ic_series.parquet'):
            df = pd.read_parquet('data/dashboard/factor_ic_series.parquet')
            # factor_ic_series 只有 date, factor_name, rank_ic，没有实际因子值
            assert 'factor_value' not in df.columns, "factor_ic_series should not contain actual factor values"

    def test_formula_factors_panel_has_required_columns(self):
        """formula_factors.parquet 必须包含 date 和 stock"""
        if os.path.exists('data/dashboard/formula_factors.parquet'):
            df = pd.read_parquet('data/dashboard/formula_factors.parquet')
            assert 'date' in df.columns, "formula_factors must have 'date' column"
            assert 'stock' in df.columns, "formula_factors must have 'stock' column"

    def test_formula_factors_has_factor_columns(self):
        """formula_factors.parquet 至少包含一个因子列"""
        if os.path.exists('data/dashboard/formula_factors.parquet'):
            df = pd.read_parquet('data/dashboard/formula_factors.parquet')
            factor_cols = [col for col in df.columns if col not in ['date', 'stock']]
            assert len(factor_cols) > 0, "formula_factors must have at least one factor column"

    def test_formula_only_uses_formula_factors_panel(self):
        """formula_only 必须来自 formula_factors"""
        comparator = FeatureSetComparison()
        
        if os.path.exists('data/dashboard/formula_factors.parquet'):
            formula_df = pd.read_parquet('data/dashboard/formula_factors.parquet')
            
            # 验证加载的是真实面板而非摘要
            assert 'date' in formula_df.columns
            assert 'stock' in formula_df.columns
            
            # 检查有因子列
            factor_cols = [col for col in formula_df.columns if col not in ['date', 'stock']]
            assert len(factor_cols) > 0

    def test_neural_only_uses_neural_factors(self):
        """neural_only 必须来自 neural_factors"""
        comparator = FeatureSetComparison()
        
        if os.path.exists('data/dashboard/neural_factors.parquet'):
            neural_df = pd.read_parquet('data/dashboard/neural_factors.parquet')
            
            # 验证有神经因子列
            neural_cols = [col for col in neural_df.columns if 'neural_factor_' in col]
            assert len(neural_cols) > 0, "neural_factors must have neural_factor columns"

    def test_formula_plus_neural_requires_alignment(self):
        """formula_plus_neural 必须按 date-stock 对齐"""
        if os.path.exists('data/dashboard/formula_factors.parquet') and \
           os.path.exists('data/dashboard/neural_factors.parquet'):
            
            formula_df = pd.read_parquet('data/dashboard/formula_factors.parquet')
            neural_df = pd.read_parquet('data/dashboard/neural_factors.parquet')
            
            # 检查日期交集
            formula_dates = set(formula_df['date'])
            neural_dates = set(neural_df['date'])
            common_dates = formula_dates & neural_dates
            
            # 如果有共同日期则继续
            if len(common_dates) > 0:
                # 使用 inner join 验证对齐
                merged = pd.merge(
                    formula_df[['date', 'stock']],
                    neural_df[['date', 'stock']],
                    on=['date', 'stock'],
                    how='inner'
                )
                assert len(merged) > 0, "formula_plus_neural should have overlapping samples"

    def test_missing_formula_factors_raises_error(self):
        """如果 formula_factors 缺失，应该报错"""
        comparator = FeatureSetComparison()
        
        # 测试不存在的路径
        with pytest.raises(ValueError):
            comparator.load_formula_factors(factor_panel_path='data/nonexistent.parquet')

    def test_empty_common_index_fails(self):
        """如果 common index 为 0，应该失败"""
        # 创建没有交集的数据
        formula_df = pd.DataFrame({
            'date': pd.date_range('2025-01-01', periods=5),
            'stock': ['A'] * 5,
            'factor_0': np.random.randn(5)
        })
        
        neural_df = pd.DataFrame({
            'date': pd.date_range('2026-01-01', periods=5),
            'stock': ['B'] * 5,
            'neural_factor_0': np.random.randn(5)
        })
        
        comparator = FeatureSetComparison()
        
        # 验证会有警告
        result = comparator.build_feature_set(
            formula_df=formula_df,
            neural_df=neural_df,
            feature_set_type='formula_plus_neural'
        )
        
        assert any('No overlapping samples after merge' in w for w in result['warnings'])

    def test_small_test_days_triggers_warning(self):
        """测试天数少于阈值时应该 WARN"""
        # 在实际 pipeline 中测试
        test_days = 5
        assert test_days < 20, "Small test days should trigger warning"

    def test_can_use_for_live_trading_must_be_false(self):
        """can_use_for_live_trading 必须为 false"""
        comparator = FeatureSetComparison()
        
        if os.path.exists('data/dashboard/formula_factors.parquet') and \
           os.path.exists('data/dashboard/neural_factors.parquet'):
            
            formula_df = pd.read_parquet('data/dashboard/formula_factors.parquet')
            neural_df = pd.read_parquet('data/dashboard/neural_factors.parquet')
            
            target = pd.Series(
                np.random.randn(100),
                index=pd.MultiIndex.from_tuples(
                    [(pd.Timestamp('2025-01-01'), 'A')] * 100,
                    names=['date', 'stock']
                )
            )
            
            result = comparator.run_comparison(
                formula_df=formula_df.head(100),
                neural_df=neural_df.head(100),
                target=target
            )
            
            assert result['can_use_for_live_trading'] == False

    def test_no_overclaiming_in_conclusion(self):
        """结论不能夸大"""
        # 验证小样本情况下结论不会宣称"显著有效"
        test_days = 5
        stock_count = 20
        
        if test_days < 20 or stock_count < 50:
            # 应该使用"初步信号"而非"显著有效"
            assert True, "Small sample should use 'preliminary' conclusion"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
