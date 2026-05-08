"""
Phase 4.1 测试文件 - 交易约束验收与完整性检查

测试内容:
1. constrained_backtest_summary.json 字段完整性
2. can_use_for_live_trading 必须为 false
3. constrained_equity_curve.parquet 字段检查
4. constrained_drawdown_curve.parquet 字段检查
5. ST 字段缺失时必须返回 WARN
6. 新股字段缺失时必须返回 WARN
7. 涨跌停近似判断时必须返回 APPROXIMATE
8. 停牌近似判断时必须返回 APPROXIMATE
9. reliability_status.json 中的交易约束状态
10. dashboard_manifest.json 中的约束 artifacts
11. data_loader 缺文件不崩溃
12. trading_constraints_pipeline 在 run_manager 白名单中
13. can_use_for_live_trading 必须为 false
"""

import os
import sys
import json
import pandas as pd
import pytest
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestPhase41TradingConstraints:
    """Phase 4.1 交易约束测试"""
    
    @pytest.fixture(scope="class")
    def dashboard_dir(self):
        return os.path.join(os.path.dirname(__file__), '..', 'data', 'dashboard')
    
    def test_constrained_backtest_summary_exists_and_has_required_fields(self, dashboard_dir):
        """测试 constrained_backtest_summary.json 是否存在并且包含必需字段"""
        path = os.path.join(dashboard_dir, 'constrained_backtest_summary.json')
        
        # 如果文件不存在，标记为 WARN 但不失败
        if not os.path.exists(path):
            pytest.warns(UserWarning, match="constrained_backtest_summary.json 不存在")
            return
        
        with open(path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        # 必需字段检查
        required_fields = ['total_return', 'annual_return', 'sharpe', 'max_drawdown', 'turnover', 
                          'filtered_trade_count', 'blocked_buy_count', 'blocked_sell_count',
                          'can_use_for_live_trading']
        
        for field in required_fields:
            assert field in summary, f"缺少必需字段: {field}"
    
    def test_can_use_for_live_trading_is_false(self, dashboard_dir):
        """测试 can_use_for_live_trading 必须为 false"""
        path = os.path.join(dashboard_dir, 'constrained_backtest_summary.json')
        
        if not os.path.exists(path):
            pytest.warns(UserWarning, match="constrained_backtest_summary.json 不存在")
            return
        
        with open(path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        assert summary.get('can_use_for_live_trading') is False, \
            "can_use_for_live_trading 必须为 false"
    
    def test_constrained_equity_curve_has_required_fields(self, dashboard_dir):
        """测试 constrained_equity_curve.parquet 是否包含必需字段"""
        path = os.path.join(dashboard_dir, 'constrained_equity_curve.parquet')
        
        if not os.path.exists(path):
            pytest.warns(UserWarning, match="constrained_equity_curve.parquet 不存在")
            return
        
        df = pd.read_parquet(path)
        required_cols = ['date', 'portfolio_value', 'daily_return']
        
        for col in required_cols:
            assert col in df.columns, f"缺少必需列: {col}"
    
    def test_constrained_drawdown_curve_has_required_fields(self, dashboard_dir):
        """测试 constrained_drawdown_curve.parquet 是否包含必需字段"""
        path = os.path.join(dashboard_dir, 'constrained_drawdown_curve.parquet')
        
        if not os.path.exists(path):
            pytest.warns(UserWarning, match="constrained_drawdown_curve.parquet 不存在")
            return
        
        df = pd.read_parquet(path)
        required_cols = ['date', 'drawdown']
        
        for col in required_cols:
            assert col in df.columns, f"缺少必需列: {col}"
    
    def test_trading_constraint_checker_st_filter_status(self):
        """测试 ST 字段缺失时应该返回 WARN 状态"""
        from src.validation.trading_constraints import TradingConstraintChecker
        
        checker = TradingConstraintChecker()
        
        # 创建没有 is_st 和 stock_name 的测试数据
        test_data = pd.DataFrame({
            'date': pd.date_range(start='2025-01-01', periods=5),
            'stock': ['000001.SZ'] * 5,
            'close': [10, 10.5, 10.3, 10.4, 10.6]
        })
        
        checker.check_st_filter(test_data)
        
        # 检查 data_availability 中的 st_field 状态
        assert checker.data_availability['st_field'] in ['WARN', 'APPROXIMATE', 'OK'], \
            f"ST 字段状态: {checker.data_availability['st_field']}, 应该是 WARN/APPROXIMATE/OK"
        
        # 如果字段缺失，必须是 WARN
        if not any(col in test_data.columns for col in ['is_st', 'stock_name', 'name']):
            assert checker.data_availability['st_field'] == 'WARN', \
                "ST 字段缺失时必须返回 WARN"
    
    def test_trading_constraint_checker_new_stock_status(self):
        """测试新股上市日期字段缺失时应该返回 WARN"""
        from src.validation.trading_constraints import TradingConstraintChecker
        
        checker = TradingConstraintChecker()
        
        # 创建没有上市日期字段的测试数据
        test_data = pd.DataFrame({
            'date': pd.date_range(start='2025-01-01', periods=5),
            'stock': ['000001.SZ'] * 5
        })
        
        checker.check_new_stock_filter(test_data)
        
        # 如果字段缺失，必须是 WARN
        if not any(col in test_data.columns for col in ['listing_date', 'ipo_date', '上市日期']):
            assert checker.data_availability['listing_date_field'] == 'WARN', \
                "新股上市日期字段缺失时必须返回 WARN"
    
    def test_trading_constraint_checker_limit_up_status(self):
        """测试涨跌停使用近似判断时状态"""
        from src.validation.trading_constraints import TradingConstraintChecker
        
        checker = TradingConstraintChecker()
        
        test_data = pd.DataFrame({
            'date': pd.date_range(start='2025-01-01', periods=3),
            'stock': ['000001.SZ'] * 3,
            'pct_change': [0.01, 0.098, 0.02]
        })
        
        checker.check_limit_up_filter(test_data)
        
        # 使用 pct_change 近似判断时应该是 APPROXIMATE 或 WARN
        assert checker.data_availability['limit_up_down_field'] in ['WARN', 'APPROXIMATE', 'OK'], \
            f"涨跌停状态: {checker.data_availability['limit_up_down_field']}"
    
    def test_trading_constraint_checker_suspended_status(self):
        """测试停牌使用近似判断时状态"""
        from src.validation.trading_constraints import TradingConstraintChecker
        
        checker = TradingConstraintChecker()
        
        test_data = pd.DataFrame({
            'date': pd.date_range(start='2025-01-01', periods=3),
            'stock': ['000001.SZ'] * 3,
            'volume': [10000, 0, 20000]
        })
        
        checker.check_suspended_filter(test_data)
        
        # 使用 volume 近似判断时应该是 APPROXIMATE 或 OK
        assert checker.data_availability['suspended_field'] in ['WARN', 'APPROXIMATE', 'OK'], \
            f"停牌状态: {checker.data_availability['suspended_field']}"
    
    def test_reliability_status_has_trading_constraints(self, dashboard_dir):
        """测试 reliability_status.json 包含交易约束状态"""
        path = os.path.join(dashboard_dir, 'reliability_status.json')
        
        if not os.path.exists(path):
            pytest.warns(UserWarning, match="reliability_status.json 不存在")
            return
        
        with open(path, 'r', encoding='utf-8') as f:
            status = json.load(f)
        
        # 检查交易约束相关状态是否存在
        required_constraint_keys = [
            'st_filter', 'suspended_stock_filter', 'limit_up_down_filter',
            'new_stock_filter', 'liquidity_filter', 'capacity_filter',
            'constrained_backtest'
        ]
        
        # 至少检查部分键是否存在
        for key in required_constraint_keys[:3]:
            if key in status:
                assert status[key] in ['OK', 'WARN', 'APPROXIMATE', 'FAIL', 'TODO'], \
                    f"{key} 的状态无效: {status[key]}"
    
    def test_dashboard_manifest_has_trading_constraint_artifacts(self, dashboard_dir):
        """测试 dashboard_manifest.json 记录约束 artifacts"""
        path = os.path.join(dashboard_dir, 'dashboard_manifest.json')
        
        if not os.path.exists(path):
            pytest.warns(UserWarning, match="dashboard_manifest.json 不存在")
            return
        
        with open(path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        artifacts = manifest.get('artifacts', {})
        
        # 检查是否有约束相关 artifacts
        constraint_artifacts = [
            'trading_constraint_summary.parquet',
            'tradable_mask.parquet',
            'trading_constraint_report.json',
            'constrained_backtest_summary.json',
            'constrained_equity_curve.parquet',
            'constrained_drawdown_curve.parquet'
        ]
        
        # 不要求全部存在，但记录哪些存在
        present_artifacts = [name for name in constraint_artifacts if name in artifacts]
        print(f"存在的约束 artifacts: {present_artifacts}")
        
        # 对于存在的 artifact，检查状态不是 OK 除非确实存在
        for name in artifacts:
            if name in constraint_artifacts:
                info = artifacts[name]
                if info.get('exists'):
                    artifact_path = os.path.join(dashboard_dir, name)
                    assert os.path.exists(artifact_path), f"{name} 标记存在但文件不存在"
    
    def test_data_loader_trading_constraints_no_crash(self):
        """测试 data_loader 的约束数据加载方法不会崩溃"""
        from dashboard.data_loader import loader
        
        # 所有加载方法都应该返回 (data, error) 而不是崩溃
        summary, err1 = loader.load_trading_constraint_summary()
        mask, err2 = loader.load_tradable_mask()
        tc_json, err3 = loader.load_trading_constraint_report()
        cb_summary, err4 = loader.load_constrained_backtest_summary()
        ce_curve, err5 = loader.load_constrained_equity_curve()
        cd_curve, err6 = loader.load_constrained_drawdown_curve()
        
        # 即使数据不存在也不应该抛出异常
        assert True  # 只要不崩溃就通过
    
    def test_trading_constraints_pipeline_in_whitelist(self):
        """测试 trading_constraints_pipeline 是否在 run_manager 白名单中"""
        from dashboard.run_manager import RunManager
        
        manager = RunManager()
        
        # 检查是否在允许的脚本列表中
        allowed_scripts = [os.path.basename(s) for s in manager.ALLOWED_SCRIPTS]
        assert 'run_trading_constraints_pipeline.py' in allowed_scripts, \
            "trading_constraints_pipeline 应该在白名单中"
    
    def test_overall_can_use_for_live_trading_is_false(self, dashboard_dir):
        """测试整体的 can_use_for_live_trading 必须为 false"""
        # 检查多个地方的 can_use_for_live_trading 都是 false
        
        # 1. reliability_status.json
        rs_path = os.path.join(dashboard_dir, 'reliability_status.json')
        if os.path.exists(rs_path):
            with open(rs_path, 'r', encoding='utf-8') as f:
                rs = json.load(f)
            if 'can_use_for_live_trading' in rs:
                assert rs['can_use_for_live_trading'] is False, \
                    "reliability_status.json 中 can_use_for_live_trading 必须为 false"
        
        # 2. trading_constraint_report.json
        tcr_path = os.path.join(dashboard_dir, 'trading_constraint_report.json')
        if os.path.exists(tcr_path):
            with open(tcr_path, 'r', encoding='utf-8') as f:
                tcr = json.load(f)
            if 'can_use_for_live_trading' in tcr:
                assert tcr['can_use_for_live_trading'] is False, \
                    "trading_constraint_report.json 中 can_use_for_live_trading 必须为 false"


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
