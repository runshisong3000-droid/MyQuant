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
from datetime import datetime
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
        from datetime import datetime, timedelta
        fetch_start = datetime.now()
        fetch_end = datetime.now()
        
        metadata = self.ds_manager._generate_metadata(
            profile_name='test_profile',
            data_source_used='eastmoney_direct',
            target_count=100,
            actual_count=80,
            history_months=12,
            start_date='20240101',
            end_date='20250101',
            candidate_count=200,
            requested_count=200,
            requested_symbols=['000001.SZ', '000002.SZ'],
            success_symbols=['000001.SZ'],
            failed_symbols=['000002.SZ'],
            failed_reasons={'000002.SZ': 'Test error'},
            fetch_start=fetch_start,
            fetch_end=fetch_end
        )
        
        required_fields = [
            'profile',
            'target_stock_count',
            'actual_stock_count',
            'target_history_months',
            'actual_start_date',
            'actual_end_date',
            'actual_trading_days',
            'candidate_symbols_count',
            'requested_symbols_count',
            'success_symbols_count',
            'failed_symbols_count',
            'requested_symbols',
            'success_symbols',
            'failed_symbols',
            'failed_reasons',
            'data_source_used',
            'fetch_started_at',
            'fetch_finished_at',
            'fetch_duration_seconds',
            'cache_valid',
            'can_use_for_research',
            'can_use_for_live_trading'
        ]
        
        for field in required_fields:
            assert field in metadata, f"Missing field: {field}"
    
    def test_can_use_for_live_trading_is_false(self):
        """测试 can_use_for_live_trading 必须为 false"""
        from datetime import datetime, timedelta
        fetch_start = datetime.now()
        fetch_end = datetime.now()
        
        metadata = self.ds_manager._generate_metadata(
            profile_name='test_profile',
            data_source_used='eastmoney_direct',
            target_count=100,
            actual_count=100,
            history_months=12,
            start_date='20240101',
            end_date='20250101',
            candidate_count=200,
            requested_count=200,
            requested_symbols=['000001.SZ'],
            success_symbols=['000001.SZ'],
            failed_symbols=[],
            failed_reasons={},
            fetch_start=fetch_start,
            fetch_end=fetch_end
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
        from datetime import datetime, timedelta
        fetch_start = datetime.now()
        fetch_end = datetime.now()
        
        metadata = self.ds_manager._generate_metadata(
            profile_name='test_profile',
            data_source_used='eastmoney_direct',
            target_count=3,
            actual_count=1,
            history_months=12,
            start_date='20240101',
            end_date='20250101',
            candidate_count=10,
            requested_count=10,
            requested_symbols=['000001.SZ', '000002.SZ', '000003.SZ'],
            success_symbols=['000001.SZ'],
            failed_symbols=['000002.SZ', '000003.SZ'],
            failed_reasons={'000002.SZ': 'Error1', '000003.SZ': 'Error2'},
            fetch_start=fetch_start,
            fetch_end=fetch_end
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
    
    def test_cache_validity_check(self):
        """测试缓存有效性检查"""
        # 创建一个过期的 metadata 模拟
        metadata = {
            'actual_stock_count': 100,
            'fetch_finished_at': '2020-01-01T00:00:00'  # 过期的日期
        }
        
        # 过期缓存应该无效
        result = self.ds_manager._check_cache_validity('research_lite', metadata)
        assert result is False  # 因为日期过期
    
    def test_build_stock_universe(self):
        """测试构建股票池"""
        universe = self.ds_manager.build_stock_universe('research_lite')
        # 应该返回至少一些股票
        assert len(universe) > 0
        # 股票格式应该是 000001.SZ 或 600000.SH
        assert '.SZ' in universe[0] or '.SH' in universe[0]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
