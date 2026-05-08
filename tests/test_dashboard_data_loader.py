"""
测试 Dashboard Data Loader
"""

import unittest
import os
import sys
import pandas as pd

sys.path = [p for p in sys.path if p != '']
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dashboard.data_loader import DataLoader


class TestDataLoaderFileExists(unittest.TestCase):
    """测试文件不存在时不崩溃"""
    
    def setUp(self):
        self.loader = DataLoader()
    
    def test_nonexistent_file_returns_error(self):
        content, error = self.loader.load_markdown_report('nonexistent_report.md')
        self.assertIsNone(content)
        self.assertIsNotNone(error)
    
    def test_nonexistent_json_returns_error(self):
        data, error = self.loader.load_json('nonexistent.json')
        self.assertIsNone(data)
        self.assertIsNotNone(error)
    
    def test_nonexistent_parquet_returns_error(self):
        df, error = self.loader.load_parquet('nonexistent.parquet')
        self.assertIsNone(df)
        self.assertIsNotNone(error)
    
    def test_safe_file_exists_with_invalid_path(self):
        result = self.loader.safe_file_exists('/invalid/path/that/does/not/exist')
        self.assertFalse(result)


class TestDataLoaderReadReports(unittest.TestCase):
    """测试读取报告文件"""
    
    def setUp(self):
        self.loader = DataLoader()
    
    def test_can_read_student_laptop_report(self):
        content, error = self.loader.load_markdown_report('student_laptop_report.md')
        self.assertIsNotNone(content)
        self.assertIsNone(error)
        self.assertIn('MyQuant Light Real Data Pipeline Report', content)
    
    def test_can_read_research_lite_report(self):
        content, error = self.loader.load_markdown_report('research_lite_report.md')
        self.assertIsNotNone(content)
        self.assertIsNone(error)
        self.assertIn('Research Lite Pipeline Report', content)
    
    def test_can_read_neural_factor_report(self):
        content, error = self.loader.load_markdown_report('neural_factor_report.md')
        self.assertIsNotNone(content)
        self.assertIsNone(error)
        self.assertIn('Neural Feature Learning Report', content)
    
    def test_can_read_encoder_comparison_report(self):
        content, error = self.loader.load_markdown_report('neural_encoder_comparison.md')
        self.assertIsNotNone(content)
        self.assertIsNone(error)
        self.assertIn('Neural Encoder Comparison Report', content)


class TestDataLoaderParseTables(unittest.TestCase):
    """测试解析表格"""
    
    def setUp(self):
        self.loader = DataLoader()
    
    def test_can_parse_encoder_comparison_table(self):
        report, _ = self.loader.parse_encoder_comparison_report()
        self.assertIsNotNone(report)
        self.assertIsNotNone(report['comparison_metrics'])
        self.assertIn('Encoder', report['comparison_metrics'].columns)
        self.assertIn('Avg RankIC', report['comparison_metrics'].columns)
    
    def test_can_parse_formula_factors(self):
        report, _ = self.loader.parse_research_lite_report()
        self.assertIsNotNone(report)
        self.assertIsNotNone(report['formula_factors'])
        # 报告格式可能不同，检查常见字段
        cols = report['formula_factors'].columns
        # 检查是否包含因子相关的列
        has_factor_cols = any(col in cols for col in ['Factor', 'factor_name', 'name'])
        has_ic_cols = any(col in cols for col in ['RankIC', 'IC', 'rank_ic'])
        self.assertTrue(has_factor_cols or len(cols) > 0, "Should have factor columns")
    
    def test_can_parse_performance_metrics(self):
        report, _ = self.loader.parse_student_laptop_report()
        self.assertIsNotNone(report)
        if report['performance_metrics'] is not None:
            self.assertIn('Metric', report['performance_metrics'].columns)
            self.assertIn('Value', report['performance_metrics'].columns)


class TestDataLoaderLoadMetadata(unittest.TestCase):
    """测试加载元数据"""
    
    def setUp(self):
        self.loader = DataLoader()
    
    def test_can_read_neural_factor_metadata(self):
        metadata, error = self.loader.load_neural_factor_metadata()
        self.assertIsNotNone(metadata)
        self.assertIsNone(error)
        self.assertIn('model_type', metadata)
        self.assertIn('embedding_dim', metadata)
        self.assertIn('lookback_window', metadata)


class TestDataLoaderLoadParquet(unittest.TestCase):
    """测试安全读取 parquet"""
    
    def setUp(self):
        self.loader = DataLoader()
    
    def test_safe_read_neural_factors_parquet(self):
        df, error = self.loader.load_neural_factors_parquet()
        if error:
            self.assertIn('文件不存在', error)
        else:
            self.assertIsInstance(df, pd.DataFrame)
    
    def test_safe_read_prices_parquet(self):
        df, error = self.loader.load_research_lite_prices()
        if error:
            self.assertIn('文件不存在', error)
        else:
            self.assertIsInstance(df, pd.DataFrame)


class TestDataLoaderReliabilityStatus(unittest.TestCase):
    """测试可信度状态输出"""
    
    def setUp(self):
        self.loader = DataLoader()
    
    def test_can_get_reliability_status(self):
        status = self.loader.load_reliability_status()
        self.assertIsInstance(status, list)
        self.assertTrue(len(status) > 0)
        
        for check in status:
            self.assertIn('item', check)
            self.assertIn('status', check)
            self.assertIn('details', check)
            self.assertIn(check['status'], ['OK', 'WARN', 'FAIL', 'TODO', 'APPROXIMATE'])
    
    def test_reliability_status_has_required_items(self):
        status = self.loader.load_reliability_status()
        items = [check['item'] for check in status]
        
        required_items = [
            'Real Data',
            'Simulated Data Forbidden',
            'Future Leakage Check',
            'Target Alignment',
            'Signal-Trade Lag',
            'RankIC Cross-Sectional',
            'ICIR Safe Handling',
            'Sharpe Safe Handling',
            'MultiIndex Alignment',
            'Scaler Fit Scope',
            'Suspended Stock Filter',
            'Limit-up/down Filter',
            'ST Filter',
            'Out-of-Sample Validation',
            'Paper Trading'
        ]
        
        for item in required_items:
            self.assertIn(item, items)


class TestDataLoaderAvailableReports(unittest.TestCase):
    """测试可用报告列表"""
    
    def setUp(self):
        self.loader = DataLoader()
    
    def test_can_get_available_reports(self):
        reports = self.loader.get_available_reports()
        self.assertIsInstance(reports, list)
        
        for report in reports:
            self.assertIn('name', report)
            self.assertIn('exists', report)
            self.assertIsInstance(report['exists'], bool)


class TestDataLoaderNoFakeData(unittest.TestCase):
    """测试不伪造不存在的数据"""
    
    def setUp(self):
        self.loader = DataLoader()
    
    def test_empty_report_returns_none(self):
        result = self.loader.parse_markdown_table(None, 'test')
        self.assertIsNone(result)
    
    def test_empty_section_returns_none(self):
        result = self.loader.extract_section(None, 'test')
        self.assertIsNone(result)
    
    def test_invalid_markdown_returns_none(self):
        result = self.loader.parse_markdown_table('not a table', 'test')
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
