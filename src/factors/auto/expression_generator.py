"""
表达式生成器 - Expression Generator

核心功能:
    - 自动生成因子表达式
    - 支持多种操作符
    - 支持窗口参数
    - 生成多样化的因子公式
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any
import random
from datetime import datetime


class ExpressionGenerator:
    """
    表达式生成器
    
    自动生成因子公式：
    基础字段：open, high, low, close, volume, amount, turnover, market_cap
    操作符：rank, zscore, delay, delta, rolling_mean, rolling_std, rolling_corr, rolling_cov, decay_linear, signed_power, log, abs
    窗口：3, 5, 10, 20, 60, 120
    """

    def __init__(self):
        self.fields = [
            'open', 'high', 'low', 'close', 'volume', 'amount', 'turnover', 'market_cap'
        ]
        
        self.unary_operators = [
            ('rank', 0),
            ('zscore', 0),
            ('log', 0),
            ('abs', 0),
            ('signed_power_2', 0),
            ('signed_power_3', 0),
            ('signed_power_-1', 0),
            ('decay_linear', 1)
        ]
        
        self.binary_operators = [
            ('delta', 1),
            ('delay', 1),
            ('rolling_mean', 1),
            ('rolling_std', 1),
            ('rolling_max', 1),
            ('rolling_min', 1),
            ('rolling_sum', 1),
            ('rolling_median', 1),
            ('rolling_skew', 1),
            ('rolling_kurt', 1)
        ]
        
        self.comparison_operators = [
            '/', '-', '+', '*'
        ]
        
        self.windows = [3, 5, 10, 20, 60, 120]
        self.decay_windows = [5, 10, 20, 30]
    
    def generate_expression(self, complexity: int = 2) -> str:
        """
        生成单个因子表达式
        
        Args:
            complexity: 表达式复杂度（1-3）
            
        Returns:
            因子表达式字符串
        """
        if complexity == 1:
            return self._generate_simple_expression()
        elif complexity == 2:
            return self._generate_medium_expression()
        else:
            return self._generate_complex_expression()
    
    def _generate_simple_expression(self) -> str:
        """生成简单表达式"""
        field = random.choice(self.fields)
        op, n_args = random.choice(self.binary_operators)
        
        if n_args == 1:
            window = random.choice(self.windows)
            return f"{op}({field}, {window})"
        else:
            return f"{op}({field})"
    
    def _generate_medium_expression(self) -> str:
        """生成中等复杂度表达式"""
        field1 = random.choice(self.fields)
        field2 = random.choice([f for f in self.fields if f != field1])
        
        op1, n_args1 = random.choice(self.binary_operators)
        op2, n_args2 = random.choice(self.unary_operators)
        
        window1 = random.choice(self.windows)
        window2 = random.choice(self.decay_windows) if op2 == 'decay_linear' else 0
        
        expr1 = f"{op1}({field1}, {window1})"
        expr2 = f"{op2}({field2}{f', {window2}' if window2 > 0 else ''})"
        
        comp_op = random.choice(self.comparison_operators)
        
        return f"{expr1} {comp_op} {expr2}"
    
    def _generate_complex_expression(self) -> str:
        """生成复杂表达式"""
        field1 = random.choice(self.fields)
        field2 = random.choice([f for f in self.fields if f != field1])
        field3 = random.choice([f for f in self.fields if f not in [field1, field2]])
        
        op1, _ = random.choice(self.binary_operators)
        op2, _ = random.choice(self.binary_operators)
        final_op, _ = random.choice(self.unary_operators)
        
        window1 = random.choice(self.windows)
        window2 = random.choice(self.windows)
        
        inner_expr = f"{op1}({field1}, {window1}) / {op2}({field2}, {window2})"
        
        if random.random() > 0.5:
            return f"{final_op}({inner_expr})"
        else:
            return f"{final_op}(corr({field1}, {field3}, {window1}))"
    
    def generate_batch(self, n_expressions: int, complexity: int = 2) -> List[str]:
        """
        批量生成表达式
        
        Args:
            n_expressions: 生成数量
            complexity: 复杂度
            
        Returns:
            表达式列表
        """
        expressions = set()
        
        while len(expressions) < n_expressions:
            expr = self.generate_expression(complexity)
            expressions.add(expr)
        
        return list(expressions)
    
    def generate_by_category(self, category: str, n_expressions: int = 10) -> List[str]:
        """
        按类别生成表达式
        
        Args:
            category: 因子类别（momentum/reversal/volatility/liquidity/value）
            n_expressions: 生成数量
            
        Returns:
            表达式列表
        """
        expressions = []
        
        for _ in range(n_expressions):
            if category == 'momentum':
                expr = self._generate_momentum_expression()
            elif category == 'reversal':
                expr = self._generate_reversal_expression()
            elif category == 'volatility':
                expr = self._generate_volatility_expression()
            elif category == 'liquidity':
                expr = self._generate_liquidity_expression()
            elif category == 'value':
                expr = self._generate_value_expression()
            else:
                expr = self.generate_expression()
            
            expressions.append(expr)
        
        return expressions
    
    def _generate_momentum_expression(self) -> str:
        """生成动量因子表达式"""
        field = random.choice(['close', 'high', 'low'])
        window1 = random.choice([5, 10, 20])
        window2 = random.choice([60, 120])
        
        return f"rank(delta(close, {window1}) / rolling_std(close, {window2}))"
    
    def _generate_reversal_expression(self) -> str:
        """生成反转因子表达式"""
        window = random.choice([1, 3, 5])
        
        return f"rank(-delta(close, {window}))"
    
    def _generate_volatility_expression(self) -> str:
        """生成波动率因子表达式"""
        window = random.choice([10, 20, 60])
        
        return f"rank(rolling_std(close, {window}))"
    
    def _generate_liquidity_expression(self) -> str:
        """生成流动性因子表达式"""
        window1 = random.choice([10, 20])
        window2 = random.choice([60, 120])
        
        return f"rank(rolling_mean(turnover, {window1}) / rolling_mean(turnover, {window2}))"
    
    def _generate_value_expression(self) -> str:
        """生成价值因子表达式"""
        return f"rank(market_cap / amount)"
    
    def evaluate_expression(self, expression: str, data: pd.DataFrame) -> pd.Series:
        """
        评估表达式并计算因子值
        
        Args:
            expression: 因子表达式
            data: 原始数据（包含open, high, low, close, volume等字段）
            
        Returns:
            因子值序列
        """
        result = None
        
        try:
            result = self._evaluate_expression(expression, data)
        except Exception as e:
            print(f"Failed to evaluate expression '{expression}': {e}")
            result = pd.Series(index=data.index)
        
        return result
    
    def _evaluate_expression(self, expression: str, data: pd.DataFrame) -> pd.Series:
        """执行表达式计算"""
        local_vars = {
            'data': data,
            'rank': self._rank,
            'zscore': self._zscore,
            'delta': self._delta,
            'delay': self._delay,
            'rolling_mean': self._rolling_mean,
            'rolling_std': self._rolling_std,
            'rolling_max': self._rolling_max,
            'rolling_min': self._rolling_min,
            'rolling_sum': self._rolling_sum,
            'rolling_median': self._rolling_median,
            'rolling_skew': self._rolling_skew,
            'rolling_kurt': self._rolling_kurt,
            'rolling_corr': self._rolling_corr,
            'rolling_cov': self._rolling_cov,
            'decay_linear': self._decay_linear,
            'signed_power_2': lambda x: x ** 2 * np.sign(x),
            'signed_power_3': lambda x: x ** 3,
            'signed_power_-1': lambda x: 1 / x,
            'log': lambda x: np.log(x.where(x > 0)),
            'abs': lambda x: abs(x),
            'corr': self._corr,
            'cov': self._cov
        }
        
        for field in self.fields:
            if field in data.columns:
                local_vars[field] = data[field]
        
        return eval(expression, {}, local_vars)
    
    def _rank(self, x: pd.Series) -> pd.Series:
        """排名"""
        return x.rank(pct=True)
    
    def _zscore(self, x: pd.Series) -> pd.Series:
        """Z-score标准化"""
        return (x - x.mean()) / x.std()
    
    def _delta(self, x: pd.Series, n: int) -> pd.Series:
        """差分"""
        return x.diff(n)
    
    def _delay(self, x: pd.Series, n: int) -> pd.Series:
        """延迟"""
        return x.shift(n)
    
    def _rolling_mean(self, x: pd.Series, window: int) -> pd.Series:
        """滚动均值"""
        return x.rolling(window).mean()
    
    def _rolling_std(self, x: pd.Series, window: int) -> pd.Series:
        """滚动标准差"""
        return x.rolling(window).std()
    
    def _rolling_max(self, x: pd.Series, window: int) -> pd.Series:
        """滚动最大值"""
        return x.rolling(window).max()
    
    def _rolling_min(self, x: pd.Series, window: int) -> pd.Series:
        """滚动最小值"""
        return x.rolling(window).min()
    
    def _rolling_sum(self, x: pd.Series, window: int) -> pd.Series:
        """滚动求和"""
        return x.rolling(window).sum()
    
    def _rolling_median(self, x: pd.Series, window: int) -> pd.Series:
        """滚动中位数"""
        return x.rolling(window).median()
    
    def _rolling_skew(self, x: pd.Series, window: int) -> pd.Series:
        """滚动偏度"""
        return x.rolling(window).skew()
    
    def _rolling_kurt(self, x: pd.Series, window: int) -> pd.Series:
        """滚动峰度"""
        return x.rolling(window).kurt()
    
    def _rolling_corr(self, x: pd.Series, y: pd.Series, window: int) -> pd.Series:
        """滚动相关性"""
        return x.rolling(window).corr(y)
    
    def _rolling_cov(self, x: pd.Series, y: pd.Series, window: int) -> pd.Series:
        """滚动协方差"""
        return x.rolling(window).cov(y)
    
    def _decay_linear(self, x: pd.Series, window: int) -> pd.Series:
        """线性衰减"""
        weights = np.arange(1, window + 1) / np.arange(1, window + 1).sum()
        return x.rolling(window).apply(lambda arr: np.dot(arr, weights))
    
    def _corr(self, x: pd.Series, y: pd.Series, window: int) -> pd.Series:
        """相关性"""
        return x.rolling(window).corr(y)
    
    def _cov(self, x: pd.Series, y: pd.Series, window: int) -> pd.Series:
        """协方差"""
        return x.rolling(window).cov(y)