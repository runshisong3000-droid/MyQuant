"""
Tests for Trading Constraints Module

测试内容:
1. ST 股票不能买入。
2. 停牌股票不能买入。
3. 停牌股票不能卖出或被标记 blocked_sell。
4. 涨停股票不能买入。
5. 跌停股票不能卖出。
6. 新股不足 N 天不能买入。
7. 成交额低于阈值不能买入。
8. 超过参与率限制必须 capacity_fail。
9. 字段缺失时返回 WARN，不返回 OK。
10. tradable_mask 字段完整。
11. constrained_backtest_summary 中 can_use_for_live_trading 必须 false。
12. 不允许实盘下单。
"""

import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.validation.trading_constraints import TradingConstraintChecker
from src.core.constrained_backtest import ConstrainedBacktestEngine


class TestTradingConstraints:
    """测试交易约束检查器"""
    
    def setup_method(self):
        """设置测试数据"""
        dates = pd.date_range(start='2025-01-01', periods=10, freq='D')
        stocks = ['000001.SZ', '000002.SZ', '000003.SZ']
        
        data = []
        for date in dates:
            for stock in stocks:
                data.append({
                    'date': date,
                    'stock': stock,
                    'open': 10.0,
                    'close': 10.0,
                    'high': 10.5,
                    'low': 9.5,
                    'volume': 10000,
                    'amount': 100000000,
                    'pct_change': 0.0
                })
        
        self.price_df = pd.DataFrame(data)
        self.config = {
            'enable_st_filter': True,
            'enable_suspended_filter': True,
            'enable_limit_up_down_filter': True,
            'enable_new_stock_filter': True,
            'enable_liquidity_filter': True,
            'enable_capacity_filter': True,
            'min_listing_days': 60,
            'min_daily_amount': 50000000,
            'max_participation_rate': 0.05,
            'default_limit_pct': 0.10
        }
        self.checker = TradingConstraintChecker(self.config)
    
    def test_st_filter_blocks_buy(self):
        """ST股票不能买入"""
        df = self.price_df.copy()
        df['is_st'] = df['stock'] == '000001.SZ'
        
        mask = self.checker.build_tradable_mask(df)
        
        # ST股票不能买入
        st_mask = mask['stock'] == '000001.SZ'
        assert mask[st_mask]['can_buy'].all() == False, "ST股票应该不能买入"
    
    def test_suspended_filter_blocks_buy(self):
        """停牌股票不能买入"""
        df = self.price_df.copy()
        df.loc[df['stock'] == '000001.SZ', 'volume'] = 0
        
        mask = self.checker.build_tradable_mask(df)
        
        suspended_mask = mask['stock'] == '000001.SZ'
        assert mask[suspended_mask]['can_buy'].all() == False, "停牌股票应该不能买入"
    
    def test_suspended_filter_blocks_sell(self):
        """停牌股票不能卖出"""
        df = self.price_df.copy()
        df.loc[df['stock'] == '000001.SZ', 'volume'] = 0
        
        mask = self.checker.build_tradable_mask(df)
        
        suspended_mask = mask['stock'] == '000001.SZ'
        assert mask[suspended_mask]['can_sell'].all() == False, "停牌股票应该不能卖出"
    
    def test_limit_up_filter_blocks_buy(self):
        """涨停股票不能买入"""
        df = self.price_df.copy()
        df.loc[df['stock'] == '000001.SZ', 'pct_change'] = 9.8
        
        mask = self.checker.build_tradable_mask(df)
        
        limit_up_mask = mask['stock'] == '000001.SZ'
        assert mask[limit_up_mask]['can_buy'].all() == False, "涨停股票应该不能买入"
    
    def test_limit_down_filter_blocks_sell(self):
        """跌停股票不能卖出"""
        df = self.price_df.copy()
        df.loc[df['stock'] == '000001.SZ', 'pct_change'] = -9.8
        
        mask = self.checker.build_tradable_mask(df)
        
        limit_down_mask = mask['stock'] == '000001.SZ'
        assert mask[limit_down_mask]['can_sell'].all() == False, "跌停股票应该不能卖出"
    
    def test_new_stock_filter_blocks_buy(self):
        """新股不足N天不能买入"""
        df = self.price_df.copy()
        df['listing_date'] = df['date'] - pd.Timedelta(days=30)
        
        mask = self.checker.build_tradable_mask(df)
        
        assert mask['can_buy'].all() == False, "新股应该不能买入"
    
    def test_liquidity_filter_blocks_buy(self):
        """成交额低于阈值不能买入"""
        df = self.price_df.copy()
        df.loc[df['stock'] == '000001.SZ', 'amount'] = 10000000
        
        mask = self.checker.build_tradable_mask(df)
        
        low_liquidity_mask = mask['stock'] == '000001.SZ'
        assert mask[low_liquidity_mask]['can_buy'].all() == False, "流动性不足的股票应该不能买入"
    
    def test_capacity_filter_blocks_buy(self):
        """超过参与率限制必须capacity_fail"""
        df = self.price_df.copy()
        df['amount'] = 100000000
        
        target_weights = {'000001.SZ': 0.5}
        mask = self.checker.build_tradable_mask(df, target_weights, portfolio_value=100000000)
        
        high_weight_mask = mask['stock'] == '000001.SZ'
        assert mask[high_weight_mask]['capacity_ok'].all() == False, "超过参与率限制应该capacity_fail"
    
    def test_missing_fields_return_warn(self):
        """字段缺失时返回WARN，不返回OK"""
        df = self.price_df.copy()
        
        mask = self.checker.build_tradable_mask(df)
        report = self.checker.generate_constraint_report(mask)
        
        # ST字段缺失应该是WARN
        assert report['data_availability']['st_field'] == 'WARN', "ST字段缺失应该返回WARN"
        # 上市日期缺失应该是WARN
        assert report['data_availability']['listing_date_field'] == 'WARN', "上市日期缺失应该返回WARN"
    
    def test_tradable_mask_has_required_columns(self):
        """tradable_mask字段完整"""
        mask = self.checker.build_tradable_mask(self.price_df)
        
        required_columns = [
            'date', 'stock', 'can_buy', 'can_sell',
            'is_st', 'is_suspended', 'is_limit_up', 'is_limit_down',
            'is_new_stock', 'liquidity_ok', 'capacity_ok', 'filtered_reason'
        ]
        
        for col in required_columns:
            assert col in mask.columns, f"tradable_mask缺少必需字段: {col}"
    
    def test_can_use_for_live_trading_is_false(self):
        """constrained_backtest_summary中can_use_for_live_trading必须false"""
        mask = self.checker.build_tradable_mask(self.price_df)
        report = self.checker.generate_constraint_report(mask)
        
        assert report['can_use_for_live_trading'] == False, "can_use_for_live_trading必须为false"
    
    def test_no_live_trading_functions(self):
        """不允许实盘下单"""
        # 检查TradingConstraintChecker没有下单相关方法
        assert 'place_order' not in dir(self.checker), "不应该有下单方法"
        assert 'execute_trade' not in dir(self.checker), "不应该有执行交易方法"
        assert 'connect_to_broker' not in dir(self.checker), "不应该有连接券商方法"


class TestConstrainedBacktest:
    """测试约束回测引擎"""
    
    def test_constrained_backtest_has_no_trading_methods(self):
        """约束回测引擎不应该有实盘下单方法"""
        engine = ConstrainedBacktestEngine()
        
        assert 'place_order' not in dir(engine), "不应该有下单方法"
        assert 'execute_trade' not in dir(engine), "不应该有执行交易方法"
        assert 'connect_to_broker' not in dir(engine), "不应该有连接券商方法"
    
    def test_save_results_has_correct_fields(self):
        """保存的结果应该包含正确字段"""
        # 验证artifacts目录结构
        output_dir = 'data/dashboard'
        
        if os.path.exists(os.path.join(output_dir, 'constrained_backtest_summary.json')):
            import json
            with open(os.path.join(output_dir, 'constrained_backtest_summary.json'), 'r') as f:
                summary = json.load(f)
            
            assert 'can_use_for_live_trading' in summary, "应该包含can_use_for_live_trading字段"
            assert summary['can_use_for_live_trading'] == False, "can_use_for_live_trading必须为false"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])