"""
TradingConstraintChecker - A股交易约束检查器

功能:
    - ST股票过滤
    - 停牌过滤
    - 涨停不可买
    - 跌停不可卖
    - 新股上市不足N天过滤
    - 流动性过滤
    - 成交容量约束
    - 构建可交易掩码
    - 生成约束审计报告

注意:
    如果数据源没有相关字段，不伪造数据，标记为WARN。
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import os


class TradingConstraintChecker:
    """
    交易约束检查器
    
    输入:
        - price_panel DataFrame
        - config 配置字典
    
    输出:
        - tradable_mask
        - constraint_summary
        - constraint_report
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._initialize_defaults()
        
    def _initialize_defaults(self):
        """初始化默认配置"""
        self.enable_st_filter = self.config.get('enable_st_filter', True)
        self.enable_suspended_filter = self.config.get('enable_suspended_filter', True)
        self.enable_limit_up_down_filter = self.config.get('enable_limit_up_down_filter', True)
        self.enable_new_stock_filter = self.config.get('enable_new_stock_filter', True)
        self.enable_liquidity_filter = self.config.get('enable_liquidity_filter', True)
        self.enable_capacity_filter = self.config.get('enable_capacity_filter', True)
        
        self.min_listing_days = self.config.get('min_listing_days', 60)
        self.min_daily_amount = self.config.get('min_daily_amount', 50000000)
        self.max_participation_rate = self.config.get('max_participation_rate', 0.05)
        self.default_limit_pct = self.config.get('default_limit_pct', 0.10)
        self.startup_board_limit_pct = self.config.get('startup_board_limit_pct', 0.20)
        self.unknown_field_policy = self.config.get('unknown_field_policy', 'WARN')
        
        # 数据可用性状态
        self.data_availability = {
            'st_field': 'WARN',
            'suspended_field': 'WARN',
            'limit_up_down_field': 'APPROXIMATE',
            'listing_date_field': 'WARN',
            'amount_field': 'OK'
        }
    
    def check_st_filter(self, price_df: pd.DataFrame) -> pd.Series:
        """
        ST股票过滤
        
        如果数据中有stock name或is_st字段，检查是否为ST股票。
        否则返回WARN状态的全False序列。
        """
        # 检查是否有ST相关字段
        has_st_field = any(col.lower() in ['is_st', 'stock_name', 'name'] for col in price_df.columns)
        
        if not has_st_field:
            self.data_availability['st_field'] = 'WARN'
            return pd.Series(False, index=price_df.index)
        
        # 如果有is_st字段，直接使用
        if 'is_st' in price_df.columns:
            self.data_availability['st_field'] = 'OK'
            return price_df['is_st'].fillna(False)
        
        # 如果有股票名称字段，检查是否包含ST
        if 'stock_name' in price_df.columns:
            self.data_availability['st_field'] = 'OK'
            return price_df['stock_name'].str.contains('ST', case=False).fillna(False)
        
        self.data_availability['st_field'] = 'WARN'
        return pd.Series(False, index=price_df.index)
    
    def check_suspended_filter(self, price_df: pd.DataFrame) -> pd.Series:
        """
        停牌过滤
        
        如果某只股票某日成交量或成交额为0，视为疑似停牌。
        """
        # 检查是否有成交量或成交额字段
        has_volume = 'volume' in price_df.columns
        has_amount = 'amount' in price_df.columns
        
        if not (has_volume or has_amount):
            self.data_availability['suspended_field'] = 'WARN'
            return pd.Series(False, index=price_df.index)
        
        self.data_availability['suspended_field'] = 'APPROXIMATE'
        
        # 成交量为0或成交额为0视为停牌
        if has_volume and has_amount:
            is_suspended = (price_df['volume'] == 0) | (price_df['amount'] == 0)
        elif has_volume:
            is_suspended = price_df['volume'] == 0
        else:
            is_suspended = price_df['amount'] == 0
        
        return is_suspended.fillna(False)
    
    def check_limit_up_filter(self, price_df: pd.DataFrame) -> pd.Series:
        """
        涨停过滤 - 涨停时不可买入
        
        使用涨跌幅近似判断是否涨停。
        普通股票约10%，创业板/科创板约20%。
        """
        if 'pct_change' not in price_df.columns:
            self.data_availability['limit_up_down_field'] = 'WARN'
            return pd.Series(False, index=price_df.index)
        
        self.data_availability['limit_up_down_field'] = 'APPROXIMATE'
        
        # 使用涨跌幅近似判断涨停（考虑四舍五入）
        is_limit_up = price_df['pct_change'] >= (self.default_limit_pct * 100 - 0.2)
        
        return is_limit_up.fillna(False)
    
    def check_limit_down_filter(self, price_df: pd.DataFrame) -> pd.Series:
        """
        跌停过滤 - 跌停时不可卖出
        
        使用涨跌幅近似判断是否跌停。
        """
        if 'pct_change' not in price_df.columns:
            self.data_availability['limit_up_down_field'] = 'WARN'
            return pd.Series(False, index=price_df.index)
        
        self.data_availability['limit_up_down_field'] = 'APPROXIMATE'
        
        # 使用涨跌幅近似判断跌停（考虑四舍五入）
        is_limit_down = price_df['pct_change'] <= -(self.default_limit_pct * 100 - 0.2)
        
        return is_limit_down.fillna(False)
    
    def check_new_stock_filter(self, price_df: pd.DataFrame) -> pd.Series:
        """
        新股过滤
        
        如果股票上市不足N个交易日，不可买入。
        如果没有上市日期字段，返回WARN状态的全False序列。
        """
        has_listing_date = any(col.lower() in ['listing_date', 'ipo_date', '上市日期'] for col in price_df.columns)
        
        if not has_listing_date:
            self.data_availability['listing_date_field'] = 'WARN'
            return pd.Series(False, index=price_df.index)
        
        self.data_availability['listing_date_field'] = 'OK'
        
        # 找到上市日期字段
        listing_col = None
        for col in price_df.columns:
            if col.lower() in ['listing_date', 'ipo_date', '上市日期']:
                listing_col = col
                break
        
        if listing_col is None:
            self.data_availability['listing_date_field'] = 'WARN'
            return pd.Series(False, index=price_df.index)
        
        # 计算上市天数
        listing_dates = pd.to_datetime(price_df[listing_col])
        days_since_listing = (price_df['date'] - listing_dates).dt.days
        
        return days_since_listing < self.min_listing_days
    
    def check_liquidity_filter(self, price_df: pd.DataFrame) -> pd.Series:
        """
        流动性过滤
        
        日成交额低于阈值的股票不可买入。
        """
        if 'amount' not in price_df.columns:
            self.data_availability['amount_field'] = 'WARN'
            return pd.Series(False, index=price_df.index)
        
        self.data_availability['amount_field'] = 'OK'
        
        return price_df['amount'] < self.min_daily_amount
    
    def check_capacity_filter(
        self,
        price_df: pd.DataFrame,
        target_weights: Optional[Dict[str, float]] = None,
        portfolio_value: float = 10000000.0
    ) -> pd.Series:
        """
        成交容量约束
        
        如果目标买入金额超过当日成交额的一定比例，标记为容量不足。
        """
        if 'amount' not in price_df.columns:
            return pd.Series(False, index=price_df.index)
        
        if target_weights is None:
            return pd.Series(False, index=price_df.index)
        
        # 计算每只股票的目标买入金额
        target_amounts = price_df['stock'].map(lambda x: target_weights.get(x, 0) * portfolio_value)
        
        # 检查是否超过参与率限制
        is_capacity_exceeded = target_amounts > (price_df['amount'] * self.max_participation_rate)
        
        return is_capacity_exceeded.fillna(False)
    
    def build_tradable_mask(
        self,
        price_panel: pd.DataFrame,
        target_weights: Optional[Dict[str, float]] = None,
        portfolio_value: float = 10000000.0
    ) -> pd.DataFrame:
        """
        构建可交易掩码
        
        Returns:
            DataFrame with columns:
                - date
                - stock
                - can_buy
                - can_sell
                - is_st
                - is_suspended
                - is_limit_up
                - is_limit_down
                - is_new_stock
                - liquidity_ok
                - capacity_ok
                - filtered_reason
        """
        df = price_panel.copy()
        
        # 初始化结果
        result = pd.DataFrame({
            'date': df['date'],
            'stock': df['stock'],
            'can_buy': True,
            'can_sell': True,
            'is_st': False,
            'is_suspended': False,
            'is_limit_up': False,
            'is_limit_down': False,
            'is_new_stock': False,
            'liquidity_ok': True,
            'capacity_ok': True,
            'filtered_reason': ''
        })
        
        # ST过滤
        if self.enable_st_filter:
            result['is_st'] = self.check_st_filter(df)
            mask = result['is_st']
            result.loc[mask, 'can_buy'] = False
            result.loc[mask, 'filtered_reason'] += 'ST股票; '
        
        # 停牌过滤
        if self.enable_suspended_filter:
            result['is_suspended'] = self.check_suspended_filter(df)
            mask = result['is_suspended']
            result.loc[mask, 'can_buy'] = False
            result.loc[mask, 'can_sell'] = False
            result.loc[mask, 'filtered_reason'] += '停牌; '
        
        # 涨停过滤（不可买）
        if self.enable_limit_up_down_filter:
            result['is_limit_up'] = self.check_limit_up_filter(df)
            mask = result['is_limit_up']
            result.loc[mask, 'can_buy'] = False
            result.loc[mask, 'filtered_reason'] += '涨停; '
        
        # 跌停过滤（不可卖）
        if self.enable_limit_up_down_filter:
            result['is_limit_down'] = self.check_limit_down_filter(df)
            mask = result['is_limit_down']
            result.loc[mask, 'can_sell'] = False
            result.loc[mask, 'filtered_reason'] += '跌停; '
        
        # 新股过滤
        if self.enable_new_stock_filter:
            result['is_new_stock'] = self.check_new_stock_filter(df)
            mask = result['is_new_stock']
            result.loc[mask, 'can_buy'] = False
            result.loc[mask, 'filtered_reason'] += '新股; '
        
        # 流动性过滤
        if self.enable_liquidity_filter:
            result['liquidity_ok'] = ~self.check_liquidity_filter(df)
            mask = ~result['liquidity_ok']
            result.loc[mask, 'can_buy'] = False
            result.loc[mask, 'filtered_reason'] += '流动性不足; '
        
        # 容量约束
        if self.enable_capacity_filter:
            result['capacity_ok'] = ~self.check_capacity_filter(df, target_weights, portfolio_value)
            mask = ~result['capacity_ok']
            result.loc[mask, 'can_buy'] = False
            result.loc[mask, 'filtered_reason'] += '容量不足; '
        
        # 清理过滤原因
        result['filtered_reason'] = result['filtered_reason'].str.strip('; ')
        
        return result
    
    def generate_constraint_report(self, tradable_mask: pd.DataFrame) -> Dict[str, Any]:
        """
        生成约束审计报告
        
        Returns:
            约束报告字典
        """
        # 按日期统计
        daily_summary = tradable_mask.groupby('date').agg(
            total_candidates=('stock', 'count'),
            tradable_count=('can_buy', 'sum'),
            st_filtered=('is_st', 'sum'),
            suspended_filtered=('is_suspended', 'sum'),
            limit_up_filtered=('is_limit_up', 'sum'),
            limit_down_filtered=('is_limit_down', 'sum'),
            new_stock_filtered=('is_new_stock', 'sum'),
            liquidity_filtered=('liquidity_ok', lambda x: (x == False).sum()),
            capacity_filtered=('capacity_ok', lambda x: (x == False).sum())
        ).reset_index()
        
        daily_summary['filtered_count'] = (
            daily_summary['st_filtered'] +
            daily_summary['suspended_filtered'] +
            daily_summary['limit_up_filtered'] +
            daily_summary['new_stock_filtered'] +
            daily_summary['liquidity_filtered'] +
            daily_summary['capacity_filtered']
        )
        
        # 总体统计
        overall_summary = {
            'total_dates': len(daily_summary),
            'average_daily_candidates': daily_summary['total_candidates'].mean(),
            'average_daily_tradable': daily_summary['tradable_count'].mean(),
            'average_daily_filtered': daily_summary['filtered_count'].mean(),
            'st_filtered_total': daily_summary['st_filtered'].sum(),
            'suspended_filtered_total': daily_summary['suspended_filtered'].sum(),
            'limit_up_filtered_total': daily_summary['limit_up_filtered'].sum(),
            'limit_down_filtered_total': daily_summary['limit_down_filtered'].sum(),
            'new_stock_filtered_total': daily_summary['new_stock_filtered'].sum(),
            'liquidity_filtered_total': daily_summary['liquidity_filtered'].sum(),
            'capacity_filtered_total': daily_summary['capacity_filtered'].sum()
        }
        
        return {
            'data_availability': self.data_availability,
            'daily_summary': daily_summary,
            'overall_summary': overall_summary,
            'generated_at': datetime.now().isoformat(),
            'can_use_for_live_trading': False
        }
    
    def save_artifacts(
        self,
        tradable_mask: pd.DataFrame,
        report: Dict[str, Any],
        output_dir: str = 'data/dashboard'
    ):
        """
        保存结构化 artifacts
        
        Args:
            tradable_mask: 可交易掩码DataFrame
            report: 约束报告
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存约束摘要
        report['daily_summary'].to_parquet(
            os.path.join(output_dir, 'trading_constraint_summary.parquet'),
            index=False
        )
        
        # 保存可交易掩码
        tradable_mask.to_parquet(
            os.path.join(output_dir, 'tradable_mask.parquet'),
            index=False
        )
        
        # 保存总体摘要
        summary_json = {
            'overall_summary': report['overall_summary'],
            'data_availability': report['data_availability'],
            'generated_at': report['generated_at'],
            'can_use_for_live_trading': report['can_use_for_live_trading']
        }
        
        with open(os.path.join(output_dir, 'trading_constraint_report.json'), 'w', encoding='utf-8') as f:
            import json
            json.dump(summary_json, f, ensure_ascii=False, indent=2)