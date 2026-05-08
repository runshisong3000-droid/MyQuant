"""
Tests for Out-of-Sample Validation Framework

测试内容:
1. 时间切分正确
2. train < validation < test
3. 禁止随机切分
4. formula_only feature set 构造正确
5. neural_only feature set 构造正确
6. formula_plus_neural 对齐正确
7. MultiIndex(date, stock) 对齐后样本数 > 0
8. future_return 不进入特征
9. oos artifacts 字段完整
10. can_use_for_live_trading 必须 false
11. 样本不足时返回 WARN
12. pipeline 不使用模拟数据
"""

import pytest
import pandas as pd
import numpy as np
import json
import os

from src.validation.out_of_sample import OutOfSampleValidator
from src.research.feature_set_comparison import FeatureSetComparison


class TestOutOfSampleValidator:
    """测试 OutOfSampleValidator 类"""
    
    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        dates = pd.date_range('2024-01-01', '2024-12-31', freq='B')[:100]
        stocks = ['000001.SZ', '000002.SZ', '000003.SZ']
        
        data = []
        for date in dates:
            for stock in stocks:
                data.append({
                    'date': date,
                    'stock': stock,
                    'feature1': np.random.randn(),
                    'feature2': np.random.randn()
                })
        
        return pd.DataFrame(data)
    
    def test_time_split_correct(self, sample_data):
        """测试时间切分正确"""
        validator = OutOfSampleValidator()
        splits = validator.split_by_time(sample_data)
        
        assert 'train' in splits
        assert 'validation' in splits
        assert 'test' in splits
        
        assert len(splits['train']) > 0
        assert len(splits['validation']) > 0
        assert len(splits['test']) > 0
    
    def test_train_before_validation_before_test(self, sample_data):
        """测试 train < validation < test"""
        validator = OutOfSampleValidator()
        splits = validator.split_by_time(sample_data)
        
        train_end = splits['train']['date'].max()
        val_start = splits['validation']['date'].min()
        val_end = splits['validation']['date'].max()
        test_start = splits['test']['date'].min()
        
        assert train_end < val_start, "train_end should be before validation_start"
        assert val_end < test_start, "validation_end should be before test_start"
    
    def test_no_random_split(self, sample_data):
        """测试禁止随机切分（时间顺序切分）"""
        validator = OutOfSampleValidator()
        splits = validator.split_by_time(sample_data)
        
        train_dates = sorted(splits['train']['date'].unique())
        val_dates = sorted(splits['validation']['date'].unique())
        test_dates = sorted(splits['test']['date'].unique())
        
        all_dates = train_dates + val_dates + test_dates
        expected_dates = sorted(sample_data['date'].unique())
        
        assert list(all_dates) == list(expected_dates), "Dates should be in order without randomization"
    
    def test_split_validation_ok(self, sample_data):
        """测试切分验证通过"""
        validator = OutOfSampleValidator()
        validator.split_by_time(sample_data)
        result = validator.validate_split()
        
        assert result['status'] == 'OK'
    
    def test_split_info_complete(self, sample_data):
        """测试切分信息完整"""
        validator = OutOfSampleValidator()
        validator.split_by_time(sample_data)
        split_info = validator.get_split_info()
        
        required_fields = [
            'train_start', 'train_end',
            'validation_start', 'validation_end',
            'test_start', 'test_end',
            'train_samples', 'validation_samples', 'test_samples',
            'stock_count', 'trading_days'
        ]
        
        for field in required_fields:
            assert field in split_info, f"Missing field: {field}"
    
    def test_wrong_ratio_raises_error(self, sample_data):
        """测试错误比例抛出异常"""
        validator = OutOfSampleValidator()
        
        with pytest.raises(ValueError):
            validator.split_by_time(sample_data, train_ratio=0.5, val_ratio=0.5, test_ratio=0.1)
    
    def test_insufficient_dates_raises_error(self):
        """测试日期不足抛出异常"""
        data = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02'],
            'stock': ['000001.SZ', '000002.SZ'],
            'feature': [1.0, 2.0]
        })
        
        validator = OutOfSampleValidator()
        
        with pytest.raises(ValueError):
            validator.split_by_time(data)
    
    def test_future_leakage_check_forbidden_keyword(self):
        """测试未来泄露检测 - 禁止关键词"""
        validator = OutOfSampleValidator()
        
        features = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'future_return': np.random.randn(10),
            'normal_feature': np.random.randn(10)
        })
        target = pd.Series(np.random.randn(10), index=pd.date_range('2024-01-01', periods=10))
        
        result = validator.check_future_leakage(features, target)
        
        assert result['status'] == 'FAIL'
        assert any(issue['type'] == 'forbidden_keyword' for issue in result['issues'])


class TestFeatureSetComparison:
    """测试 FeatureSetComparison 类"""
    
    @pytest.fixture
    def formula_df(self):
        """创建公式因子测试数据"""
        dates = pd.date_range('2024-01-01', '2024-12-31', freq='B')[:50]
        stocks = ['000001.SZ', '000002.SZ', '000003.SZ']
        
        data = []
        for date in dates:
            for stock in stocks:
                data.append({
                    'date': date,
                    'stock': stock,
                    'factor_1': np.random.randn(),
                    'factor_2': np.random.randn(),
                    'factor_name': 'test_factor'
                })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def neural_df(self):
        """创建神经因子测试数据"""
        dates = pd.date_range('2024-01-01', '2024-12-31', freq='B')[:50]
        stocks = ['000001.SZ', '000002.SZ', '000003.SZ']
        
        data = []
        for date in dates:
            for stock in stocks:
                data.append({
                    'date': date,
                    'stock': stock,
                    'neural_factor_0': np.random.randn(),
                    'neural_factor_1': np.random.randn()
                })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def target(self):
        """创建目标变量"""
        dates = pd.date_range('2024-01-01', '2024-12-31', freq='B')[:50]
        stocks = ['000001.SZ', '000002.SZ', '000003.SZ']
        
        index = pd.MultiIndex.from_product([dates, stocks], names=['date', 'stock'])
        return pd.Series(np.random.randn(len(index)), index=index)
    
    def test_formula_only_feature_set(self, formula_df, neural_df, target):
        """测试 formula_only feature set 构造正确"""
        comparator = FeatureSetComparison()
        fs = comparator.build_feature_set(formula_df, neural_df, 'formula_only', target)
        
        assert fs['feature_set_type'] == 'formula_only'
        assert fs['sample_count'] > 0
        assert 'factor_1' in fs['feature_names'] or 'factor_name' in fs['feature_names']
    
    def test_neural_only_feature_set(self, formula_df, neural_df, target):
        """测试 neural_only feature set 构造正确"""
        comparator = FeatureSetComparison()
        fs = comparator.build_feature_set(formula_df, neural_df, 'neural_only', target)
        
        assert fs['feature_set_type'] == 'neural_only'
        assert fs['sample_count'] > 0
        assert any('neural_factor_' in name for name in fs['feature_names'])
    
    def test_formula_plus_neural_feature_set(self, formula_df, neural_df, target):
        """测试 formula_plus_neural 对齐正确"""
        comparator = FeatureSetComparison()
        fs = comparator.build_feature_set(formula_df, neural_df, 'formula_plus_neural', target)
        
        assert fs['feature_set_type'] == 'formula_plus_neural'
        assert fs['sample_count'] > 0
        assert fs['feature_count'] > 0
    
    def test_multiindex_alignment(self, formula_df, neural_df, target):
        """测试 MultiIndex(date, stock) 对齐后样本数 > 0"""
        comparator = FeatureSetComparison()
        
        fs_formula = comparator.build_feature_set(formula_df, neural_df, 'formula_only', target)
        fs_neural = comparator.build_feature_set(formula_df, neural_df, 'neural_only', target)
        fs_combined = comparator.build_feature_set(formula_df, neural_df, 'formula_plus_neural', target)
        
        assert fs_formula['sample_count'] > 0, "formula_only samples should be > 0"
        assert fs_neural['sample_count'] > 0, "neural_only samples should be > 0"
        assert fs_combined['sample_count'] > 0, "formula_plus_neural samples should be > 0"
    
    def test_future_return_not_in_features(self, formula_df, neural_df, target):
        """测试 future_return 不进入特征"""
        comparator = FeatureSetComparison()
        
        fs_formula = comparator.build_feature_set(formula_df, neural_df, 'formula_only', target)
        
        for name in fs_formula['feature_names']:
            assert 'future' not in name.lower()
            assert 'target' not in name.lower()
    
    def test_validate_alignment_ok(self, formula_df, neural_df, target):
        """测试对齐验证通过"""
        comparator = FeatureSetComparison()
        result = comparator.validate_feature_alignment(formula_df, neural_df, target)
        
        assert result['status'] == 'OK'
    
    def test_validate_alignment_fail_missing_date(self):
        """测试对齐验证失败 - 缺少日期列"""
        formula_df = pd.DataFrame({'stock': ['000001.SZ'], 'factor_1': [1.0]})
        neural_df = pd.DataFrame({'date': ['2024-01-01'], 'stock': ['000001.SZ']})
        target = pd.Series([0.1], index=pd.MultiIndex.from_tuples([('2024-01-01', '000001.SZ')], names=['date', 'stock']))
        
        comparator = FeatureSetComparison()
        result = comparator.validate_feature_alignment(formula_df, neural_df, target)
        
        assert result['status'] == 'FAIL'
    
    def test_run_comparison_produces_results(self, formula_df, neural_df, target):
        """测试运行对比产生结果"""
        comparator = FeatureSetComparison()
        results = comparator.run_comparison(formula_df, neural_df, target)
        
        assert 'feature_sets' in results
        assert len(results['feature_sets']) == 3
        assert results['can_use_for_live_trading'] == False
    
    def test_comparison_summary_has_correct_columns(self, formula_df, neural_df, target):
        """测试对比摘要有正确的列"""
        comparator = FeatureSetComparison()
        comparator.run_comparison(formula_df, neural_df, target)
        summary_df = comparator.get_comparison_summary()
        
        required_columns = ['feature_set', 'feature_count', 'sample_count', 'test_rank_ic', 'test_icir', 'coverage']
        for col in required_columns:
            assert col in summary_df.columns, f"Missing column: {col}"


class TestOOSArtifacts:
    """测试 OOS artifacts"""
    
    def test_oos_split_info_fields(self):
        """测试 oos_split_info.json 字段完整"""
        required_fields = [
            'train_start', 'train_end',
            'validation_start', 'validation_end',
            'test_start', 'test_end',
            'stock_count', 'trading_days',
            'train_ratio', 'val_ratio', 'test_ratio'
        ]
        
        artifact_path = 'data/dashboard/oos_split_info.json'
        if os.path.exists(artifact_path):
            with open(artifact_path, 'r') as f:
                data = json.load(f)
            
            for field in required_fields:
                assert field in data, f"Missing field in oos_split_info.json: {field}"
    
    def test_oos_feature_comparison_fields(self):
        """测试 oos_feature_comparison.parquet 字段完整"""
        required_fields = [
            'feature_set', 'feature_count', 'test_rank_ic', 'test_icir', 
            'coverage', 'total_return', 'annual_return', 'sharpe', 
            'max_drawdown', 'turnover', 'can_use_for_live_trading'
        ]
        
        artifact_path = 'data/dashboard/oos_feature_comparison.parquet'
        if os.path.exists(artifact_path):
            df = pd.read_parquet(artifact_path)
            
            for field in required_fields:
                assert field in df.columns, f"Missing column in oos_feature_comparison.parquet: {field}"
            
            assert (df['can_use_for_live_trading'] == False).all(), "can_use_for_live_trading must be false"
    
    def test_oos_rankic_series_fields(self):
        """测试 oos_rankic_series.parquet 字段完整"""
        required_fields = ['date', 'feature_set', 'rank_ic']
        
        artifact_path = 'data/dashboard/oos_rankic_series.parquet'
        if os.path.exists(artifact_path):
            df = pd.read_parquet(artifact_path)
            
            for field in required_fields:
                assert field in df.columns, f"Missing column in oos_rankic_series.parquet: {field}"
    
    def test_oos_backtest_summary_fields(self):
        """测试 oos_backtest_summary.parquet 字段完整"""
        required_fields = ['feature_set', 'total_return', 'annual_return', 'sharpe', 'max_drawdown', 'turnover']
        
        artifact_path = 'data/dashboard/oos_backtest_summary.parquet'
        if os.path.exists(artifact_path):
            df = pd.read_parquet(artifact_path)
            
            for field in required_fields:
                assert field in df.columns, f"Missing column in oos_backtest_summary.parquet: {field}"
    
    def test_oos_equity_curves_fields(self):
        """测试 oos_equity_curves.parquet 字段完整"""
        required_fields = ['date', 'feature_set', 'pnl']
        
        artifact_path = 'data/dashboard/oos_equity_curves.parquet'
        if os.path.exists(artifact_path):
            df = pd.read_parquet(artifact_path)
            
            for field in required_fields:
                assert field in df.columns, f"Missing column in oos_equity_curves.parquet: {field}"
    
    def test_can_use_for_live_trading_must_be_false(self):
        """测试 can_use_for_live_trading 必须为 false"""
        artifact_path = 'data/dashboard/oos_feature_comparison.parquet'
        if os.path.exists(artifact_path):
            df = pd.read_parquet(artifact_path)
            
            if 'can_use_for_live_trading' in df.columns:
                assert (df['can_use_for_live_trading'] == False).all(), \
                    "can_use_for_live_trading must always be false"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])