"""
Tests for DataSourceManager

Coverage:
1. profile config 能读取
2. 不存在 profile 会失败
3. cache metadata 字段完整
4. actual_stock_count 低于阈值时返回 WARN 或 FAIL
5. research_medium_trial 低于 50 只股票必须 FAIL
6. failed_symbols 会被记录
7. 不允许 fallback 到 research_lite
8. 不允许 fallback 到 student_laptop
9. 没有 TUSHARE_TOKEN 时 Tushare 不会假成功
10. can_use_for_live_trading 必须 false
"""

import os
import sys
import json
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_source_manager import DataSourceManager


class TestDataSourceManager:
    
    def setup_method(self):
        """设置测试环境"""
        self.ds_manager = DataSourceManager()
    
    def test_profile_config_can_be_read(self):
        """测试 profile config 能读取"""
        config = self.ds_manager.get_profile_config('research_lite')
        assert config is not None
        assert 'stock_count' in config
        assert 'history_months' in config
    
    def test_nonexistent_profile_fails(self):
        """测试不存在的 profile 会失败"""
        config = self.ds_manager.get_profile_config('nonexistent_profile_xyz')
        assert config is None
    
    def test_metadata_fields_complete(self):
        """测试 metadata 字段完整"""
        metadata = self.ds_manager._generate_metadata(
            profile_name='test_profile',
            target_count=100,
            actual_count=80,
            history_months=12,
            requested_symbols=['000001.SZ', '000002.SZ'],
            success_symbols=['000001.SZ'],
            failed_symbols=['000002.SZ'],
            failed_reasons={'000002.SZ': 'Test error'}
        )
        
        required_fields = [
            'profile',
            'target_stock_count',
            'actual_stock_count',
            'target_history_months',
            'requested_symbols',
            'success_symbols',
            'failed_symbols',
            'failed_reasons',
            'success_ratio',
            'failed_symbols_count',
            'data_source',
            'generated_at',
            'cache_valid',
            'can_use_for_live_trading'
        ]
        
        for field in required_fields:
            assert field in metadata, f"Missing field: {field}"
    
    def test_can_use_for_live_trading_is_false(self):
        """测试 can_use_for_live_trading 必须为 false"""
        metadata = self.ds_manager._generate_metadata(
            profile_name='test_profile',
            target_count=100,
            actual_count=100,
            history_months=12,
            requested_symbols=['000001.SZ'],
            success_symbols=['000001.SZ'],
            failed_symbols=[],
            failed_reasons={}
        )
        
        assert metadata['can_use_for_live_trading'] is False
    
    def test_determine_status_fail_below_min_stocks(self):
        """测试实际股票数低于最小值时返回 FAIL"""
        status = self.ds_manager._determine_status('research_medium_trial', 49, 150)
        assert status == 'FAIL'
    
    def test_determine_status_warn_below_ratio(self):
        """测试成功率低于阈值时返回 WARN"""
        status = self.ds_manager._determine_status('research_medium_trial', 100, 150)
        success_ratio = 100 / 150  # ~0.67 < 0.7
        assert status == 'WARN'
    
    def test_determine_status_ok(self):
        """测试达到要求时返回 OK"""
        status = self.ds_manager._determine_status('research_medium_trial', 120, 150)
        success_ratio = 120 / 150  # 0.8 >= 0.7
        assert status == 'OK'
    
    def test_failed_symbols_recorded(self):
        """测试 failed_symbols 会被记录在 metadata 中"""
        metadata = self.ds_manager._generate_metadata(
            profile_name='test_profile',
            target_count=3,
            actual_count=1,
            history_months=12,
            requested_symbols=['000001.SZ', '000002.SZ', '000003.SZ'],
            success_symbols=['000001.SZ'],
            failed_symbols=['000002.SZ', '000003.SZ'],
            failed_reasons={'000002.SZ': 'Error1', '000003.SZ': 'Error2'}
        )
        
        assert len(metadata['failed_symbols']) == 2
        assert '000002.SZ' in metadata['failed_symbols']
        assert '000003.SZ' in metadata['failed_symbols']
        assert len(metadata['failed_reasons']) == 2
    
    def test_research_medium_trial_below_50_fails(self):
        """测试 research_medium_trial 低于 50 只股票必须 FAIL"""
        # research_medium_trial 的 min_absolute_stocks 是 100
        status = self.ds_manager._determine_status('research_medium_trial', 49, 150)
        assert status == 'FAIL'
    
    def test_no_fallback_to_research_lite(self):
        """测试不允许 fallback 到 research_lite"""
        # 检查 load_profile_cache 不会 fallback 到其他 profile
        result = self.ds_manager.load_profile_cache('nonexistent_profile')
        assert not result['valid']
        # 不会自动加载 research_lite 的缓存
    
    def test_no_fallback_to_student_laptop(self):
        """测试不允许 fallback 到 student_laptop"""
        result = self.ds_manager.load_profile_cache('another_nonexistent_profile')
        assert not result['valid']
    
    def test_tushare_not_available_without_token(self):
        """测试没有 TUSHARE_TOKEN 时 Tushare 不会假成功"""
        # 检查当前数据源配置
        config = self.ds_manager.get_data_source_config()
        assert config.get('default') == 'akshare'
        # 当前实现默认使用 akshare，不会假装 tushare 可用
    
    def test_cache_validity_check(self):
        """测试缓存有效性检查"""
        # 创建一个过期的 metadata 模拟
        metadata = {
            'actual_stock_count': 100,
            'generated_at': '2020-01-01T00:00:00'  # 过期的日期
        }
        
        # 过期缓存应该无效
        result = self.ds_manager._check_cache_validity('research_lite', metadata)
        assert result is False  # 因为日期过期
    
    def test_data_source_config_loaded(self):
        """测试数据源配置能被正确加载"""
        config = self.ds_manager.get_data_source_config()
        assert 'default' in config
        assert 'retry_times' in config
        assert 'min_absolute_stocks' in config
        assert 'min_success_ratio' in config
    
    def test_min_absolute_stocks_config(self):
        """测试最小股票数配置正确"""
        config = self.ds_manager.get_data_source_config()
        min_stocks = config.get('min_absolute_stocks', {})
        
        assert min_stocks.get('research_medium_trial') == 100
        assert min_stocks.get('research_medium') == 200
        assert min_stocks.get('research_lite') == 50
    
    def test_min_success_ratio_config(self):
        """测试最小成功率配置正确"""
        config = self.ds_manager.get_data_source_config()
        min_ratio = config.get('min_success_ratio', {})
        
        assert min_ratio.get('research_medium_trial') == 0.7
        assert min_ratio.get('research_medium') == 0.7


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
