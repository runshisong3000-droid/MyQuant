"""
增强版因子生成器 - Enhanced Expression Generator

核心功能:
    - 自动生成200+候选因子
    - 动量类因子
    - 反转类因子
    - 波动率类因子
    - 流动性类因子
    - 量价类因子
    - 基础因子×不同窗口×中性化
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any
from datetime import datetime
import itertools


class EnhancedFactorGenerator:
    """
    增强版因子生成器
    
    批量生成200+候选因子：
    1. 基础因子（50个）
    2. 不同窗口扩展（×5）
    3. 中性化/标准化处理（×2）
    总计：50 × 5 × 2 = 500+ 候选因子
    """

    def __init__(self):
        self.windows = [3, 5, 10, 20, 60]
        self.shorts = [1, 3, 5]
        self.longs = [10, 20, 60]
        
        self.factors = []
        self.factor_info = {}
    
    def generate_all_factors(
        self,
        price_data: pd.DataFrame,
        generate_neutral: bool = True
    ) -> Dict[str, pd.Series]:
        """
        生成所有候选因子
        
        Args:
            price_data: 价格数据（包含open, high, low, close, volume, amount, turnover）
            generate_neutral: 是否生成中性化/标准化版本
            
        Returns:
            因子字典
        """
        self.factors = {}
        
        # 生成动量类因子
        self._generate_momentum_factors(price_data)
        
        # 生成反转类因子
        self._generate_reversal_factors(price_data)
        
        # 生成波动率类因子
        self._generate_volatility_factors(price_data)
        
        # 生成流动性类因子
        self._generate_liquidity_factors(price_data)
        
        # 生成量价类因子
        self._generate_price_volume_factors(price_data)
        
        # 生成技术指标类因子
        self._generate_technical_factors(price_data)
        
        # 生成基础因子扩展
        self._generate_basic_factors(price_data)
        
        # 生成中性化/标准化版本
        if generate_neutral:
            self._generate_neutralized_factors()
        
        print(f'Generated {len(self.factors)} candidate factors')
        return self.factors
    
    def _generate_momentum_factors(self, data: pd.DataFrame):
        """生成动量类因子"""
        # 简单动量
        for window in self.windows:
            if 'close' in data.columns:
                name = f'momentum_simple_{window}'
                self._add_factor(name, self._simple_momentum(data['close'], window), 'momentum')
        
        # 相对动量（vs 20日平均）
        for window in self.windows:
            if 'close' in data.columns:
                name = f'momentum_rel_ma{window}'
                self._add_factor(name, self._relative_momentum(data['close'], window), 'momentum')
        
        # 动量的变化率
        for long_window in self.longs:
            for short_window in self.shorts:
                if 'close' in data.columns and long_window > short_window:
                    name = f'momentum_chg_{short_window}vs{long_window}'
                    self._add_factor(name, self._momentum_change(data['close'], short_window, long_window), 'momentum')
    
    def _generate_reversal_factors(self, data: pd.DataFrame):
        """生成反转类因子"""
        # 短期反转
        for window in self.shorts:
            if 'close' in data.columns:
                name = f'reversal_short_{window}'
                self._add_factor(name, -self._simple_momentum(data['close'], window), 'reversal')
        
        # 开盘-收盘反转
        if 'open' in data.columns and 'close' in data.columns:
            name = 'reversal_overnight'
            self._add_factor(name, (data['open'] - data['close'].shift(1)) / data['close'].shift(1), 'reversal')
        
        # 日内反转
        if 'open' in data.columns and 'close' in data.columns:
            name = 'reversal_intraday'
            self._add_factor(name, (data['close'] - data['open']) / data['open'], 'reversal')
    
    def _generate_volatility_factors(self, data: pd.DataFrame):
        """生成波动率类因子"""
        # 收益率波动率
        for window in self.windows:
            if 'close' in data.columns:
                name = f'volatility_std_{window}'
                returns = data['close'].pct_change()
                self._add_factor(name, returns.rolling(window).std(), 'volatility')
        
        # 日内波动率
        if 'high' in data.columns and 'low' in data.columns:
            for window in self.windows:
                name = f'volatility_intraday_{window}'
                self._add_factor(name, (data['high'] - data['low']).rolling(window).std(), 'volatility')
        
        # 下行波动率
        for window in self.windows:
            if 'close' in data.columns:
                name = f'volatility_downside_{window}'
                returns = data['close'].pct_change()
                downside = returns.where(returns < 0, 0)
                self._add_factor(name, downside.rolling(window).std(), 'volatility')
    
    def _generate_liquidity_factors(self, data: pd.DataFrame):
        """生成流动性类因子"""
        # 成交额
        if 'amount' in data.columns:
            for window in self.windows:
                name = f'liquidity_amount_ma{window}'
                self._add_factor(name, data['amount'].rolling(window).mean(), 'liquidity')
        
        # 换手率
        if 'turnover' in data.columns:
            for window in self.windows:
                name = f'liquidity_turnover_ma{window}'
                self._add_factor(name, data['turnover'].rolling(window).mean(), 'liquidity')
        
        # 换手率变化
        if 'turnover' in data.columns:
            for window in self.windows:
                name = f'liquidity_turnover_chg_{window}'
                self._add_factor(name, data['turnover'] / data['turnover'].rolling(window).mean(), 'liquidity')
    
    def _generate_price_volume_factors(self, data: pd.DataFrame):
        """生成量价类因子"""
        # 量价背离
        if 'close' in data.columns and 'volume' in data.columns:
            for window in self.windows:
                name = f'pv_corr_{window}'
                price_ret = data['close'].pct_change()
                vol_ret = data['volume'].pct_change()
                self._add_factor(name, price_ret.rolling(window).corr(vol_ret), 'price_volume')
        
        # 放量上涨/缩量下跌
        if 'close' in data.columns and 'volume' in data.columns:
            for window in self.windows:
                name = f'pv_vol_price_corr_{window}'
                self._add_factor(name, data['close'].rolling(window).corr(data['volume']), 'price_volume')
        
        # 成交量分位数
        if 'volume' in data.columns:
            for window in self.windows:
                name = f'volume_percentile_{window}'
                self._add_factor(name, data['volume'].rolling(window).rank(pct=True), 'price_volume')
    
    def _generate_technical_factors(self, data: pd.DataFrame):
        """生成技术指标类因子"""
        # RSI-like
        if 'close' in data.columns:
            for window in self.windows:
                name = f'tech_rsi_{window}'
                returns = data['close'].pct_change()
                up = returns.where(returns > 0, 0)
                down = -returns.where(returns < 0, 0)
                rsi = 100 - 100 / (1 + up.rolling(window).mean() / down.rolling(window).mean())
                self._add_factor(name, rsi, 'technical')
        
        # MACD-like (差)
        if 'close' in data.columns:
            for fast_window, slow_window in [(5, 20), (10, 60)]:
                name = f'tech_macd_{fast_window}vs{slow_window}'
                macd = data['close'].rolling(fast_window).mean() - data['close'].rolling(slow_window).mean()
                self._add_factor(name, macd, 'technical')
        
        # 布林带位置
        if 'close' in data.columns:
            for window in self.windows:
                name = f'tech_bb_pos_{window}'
                ma = data['close'].rolling(window).mean()
                std = data['close'].rolling(window).std()
                z_score = (data['close'] - ma) / (std + 1e-8)
                self._add_factor(name, z_score, 'technical')
    
    def _generate_basic_factors(self, data: pd.DataFrame):
        """生成基础因子"""
        # 基于基础数据的简单变换
        base_ops = {
            'rank': lambda x: x.rank(pct=True),
            'zscore': lambda x: (x - x.mean()) / x.std(),
            'log': lambda x: np.log(x.where(x > 0, np.nan)),
        }
        
        # 对每个基础数据应用变换
        for col in data.columns:
            for op_name, op_func in base_ops.items():
                try:
                    name = f'basic_{col}_{op_name}'
                    self._add_factor(name, op_func(data[col]), 'basic')
                except:
                    pass
    
    def _generate_neutralized_factors(self):
        """生成中性化/标准化版本"""
        factor_names = list(self.factors.keys())
        
        for name in factor_names:
            data = self.factors[name]
            
            # 标准化版本
            std_name = f'{name}_std'
            try:
                std_data = (data - data.mean()) / (data.std() + 1e-8)
                self._add_factor(std_name, std_data, 'neutralized')
            except:
                pass
            
            # 排名版本
            rank_name = f'{name}_rank'
            try:
                if isinstance(data.index, pd.MultiIndex):
                    rank_data = data.groupby(level=0).rank(pct=True)
                else:
                    rank_data = data.rank(pct=True)
                self._add_factor(rank_name, rank_data, 'neutralized')
            except:
                pass
    
    def _simple_momentum(self, close: pd.Series, window: int) -> pd.Series:
        """简单动量"""
        return close.pct_change(window)
    
    def _relative_momentum(self, close: pd.Series, window: int) -> pd.Series:
        """相对动量（vs 移动平均）"""
        return close / close.rolling(window).mean() - 1
    
    def _momentum_change(self, close: pd.Series, short_window: int, long_window: int) -> pd.Series:
        """动量变化率"""
        short_mom = close.pct_change(short_window)
        long_mom = close.pct_change(long_window)
        return short_mom - long_mom
    
    def _add_factor(self, name: str, data: pd.Series, category: str):
        """添加因子"""
        if data is not None and not data.isna().all():
            self.factors[name] = data
            self.factor_info[name] = {
                'category': category,
                'added_at': datetime.now().isoformat()
            }
    
    def get_factor_categories(self) -> Dict[str, List[str]]:
        """获取因子分类"""
        categories = {}
        for name, info in self.factor_info.items():
            cat = info['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(name)
        return categories
    
    def get_summary(self) -> pd.DataFrame:
        """获取因子摘要"""
        summary = []
        for name in self.factors.keys():
            data = self.factors[name]
            summary.append({
                'name': name,
                'category': self.factor_info[name]['category'],
                'coverage': 1 - data.isna().mean(),
                'n_samples': len(data.dropna())
            })
        return pd.DataFrame(summary)
