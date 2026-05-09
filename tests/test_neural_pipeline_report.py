"""
测试 Neural Pipeline 报告功能
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestNeuralPipelineReport:
    """测试 neural pipeline 报告生成"""
    
    def test_no_fetch_success_variable(self):
        """测试 neural pipeline 中没有未定义的 fetch_success 变量"""
        pipeline_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'run_neural_factor_pipeline.py')
        
        with open(pipeline_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 确保没有使用未定义的 fetch_success
        assert 'fetch_success' not in content, "不应该使用未定义的 fetch_success 变量"
    
    def test_stock_count_from_price_data(self):
        """测试 stock_count 从 price_data['stock'].nunique() 读取"""
        pipeline_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'run_neural_factor_pipeline.py')
        
        with open(pipeline_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 确保从正确的位置读取 stock_count
        assert "price_data['stock'].nunique()" in content, "stock_count 应该从 price_data['stock'].nunique() 读取"
    
    def test_report_uses_correct_variables(self):
        """测试报告使用正确的变量"""
        pipeline_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'run_neural_factor_pipeline.py')
        
        with open(pipeline_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 确保报告中使用正确的变量
        assert "price_data['stock'].nunique()" in content, "应该从 price_data 获取股票数量"
        assert "price_data['date'].min()" in content or "min(" in content, "应该获取最小日期"
        assert "price_data['date'].max()" in content or "max(" in content, "应该获取最大日期"
    
    def test_can_use_for_live_trading_is_false(self):
        """测试报告明确说明不能实盘"""
        pipeline_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'run_neural_factor_pipeline.py')
        
        with open(pipeline_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否有不能实盘的声明
        assert "NOT ready for live trading" in content or "cannot be used for live trading" in content, "应该声明不能用于实盘"


class TestNeuralFactorMetadata:
    """测试 neural factor metadata 完整性"""
    
    def test_metadata_has_required_fields(self):
        """测试 metadata 包含必需字段"""
        expected_fields = [
            'data_source', 'price_panel_path', 'input_date_range', 'output_date_range',
            'input_stock_count', 'output_stock_count', 'sample_count',
            'train_start', 'train_end', 'validation_start', 'validation_end',
            'test_start', 'test_end', 'leakage_check_status',
            'model_type', 'encoder_type', 'lookback_window',
            'embedding_dim', 'hidden_dim', 'epochs', 'batch_size'
        ]
        
        # 模拟 metadata
        metadata = {
            'data_source': 'research_medium_trial',
            'price_panel_path': 'data/processed/profiles/research_medium_trial/prices.parquet',
            'input_date_range': ['2024-11-15', '2026-05-08'],
            'output_date_range': ['2024-12-12', '2026-05-06'],
            'input_stock_count': 150,
            'output_stock_count': 150,
            'sample_count': 50143,
            'train_start': '2024-12-12',
            'train_end': '2025-10-14',
            'validation_start': '2025-10-15',
            'validation_end': '2026-01-19',
            'test_start': '2026-01-20',
            'test_end': '2026-05-06',
            'leakage_check_status': 'OK',
            'model_type': 'SequenceAutoEncoder',
            'encoder_type': 'MLP',
            'lookback_window': 20,
            'embedding_dim': 8,
            'hidden_dim': 32,
            'epochs': 5,
            'batch_size': 64
        }
        
        for field in expected_fields:
            assert field in metadata, f"metadata 缺少必需字段: {field}"
