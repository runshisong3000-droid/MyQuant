"""
轻量测试脚本 - 验证核心功能
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.data.data_provider import MockDataProvider
from src.factors.auto.enhanced_generator import EnhancedFactorGenerator
from src.factors.auto.factor_evaluator import FactorEvaluator
from src.core.backtest import CrossSectionBacktestEngine

print("="*50)
print("MyQuant 核心功能测试")
print("="*50)

print("\n[1] 测试数据提供...")
provider = MockDataProvider(n_stocks=30, n_days=200)
features = provider.get_all_features()
returns = provider.get_returns()
print(f"  数据: {features.shape}")
print(f"  日期: {features.index.get_level_values(0).min()} ~ {features.index.get_level_values(0).max()}")

print("\n[2] 测试因子生成...")
gen = EnhancedFactorGenerator()
factors = gen.generate_all_factors(features, generate_neutral=False)
print(f"  生成因子: {len(factors)} 个")

print("\n[3] 测试因子评估...")
evaluator = FactorEvaluator()
test_factor = list(factors.values())[0]
eval_result = evaluator.evaluate_single(test_factor, returns)
print(f"  RankIC: {eval_result['rank_ic']['mean']:.4f}")
print(f"  ICIR: {eval_result.get('icir', 0):.4f}")
print(f"  覆盖率: {eval_result.get('coverage', 0):.2f}")

print("\n[4] 测试回测引擎...")
engine = CrossSectionBacktestEngine()
bt_result = engine.run(test_factor, returns, n_groups=10, long_short=True)
print(f"  总收益: {bt_result['total_return']*100:.2f}%")
print(f"  Sharpe: {bt_result['sharpe_ratio']:.2f}")
print(f"  最大回撤: {bt_result['max_drawdown']*100:.2f}%")

print("\n" + "="*50)
print("核心功能测试通过!")
print("="*50)
