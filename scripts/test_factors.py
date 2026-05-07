#!/usr/bin/env python
"""测试因子计算功能"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.factors import TechnicalFactorEngine, FactorAnalyzer


def test_technical_factors():
    """测试技术因子计算"""
    print("\n" + "=" * 60)
    print("Testing Technical Factors")
    print("=" * 60)

    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    base = 100
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = base * (1 + returns).cumprod()

    data = pd.DataFrame({
        'open': prices * (1 - np.random.normal(0.005, 0.01, len(dates))),
        'high': prices * (1 + np.random.normal(0.01, 0.005, len(dates))),
        'low': prices * (1 - np.random.normal(0.01, 0.005, len(dates))),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)

    engine = TechnicalFactorEngine()

    print("\nComputing all technical factors...")
    factors = engine.compute_all(data)

    print(f"\nGenerated {len(factors.columns)} factors:")
    print(factors.columns.tolist())

    print(f"\nFactor shape: {factors.shape}")
    print(f"Non-null ratio:")
    non_null_ratio = (factors.count() / len(factors) * 100).round(1)
    print(non_null_ratio)

    print("\nSample factor values:")
    print(factors[['rsi', 'atr', 'obv', 'adx']].head())

    print("\n" + "=" * 60)
    print("Technical Factors Test PASSED")
    print("=" * 60)

    return factors


def test_factor_analysis(factors):
    """测试因子分析"""
    print("\n" + "=" * 60)
    print("Testing Factor Analysis")
    print("=" * 60)

    if factors is None or len(factors) == 0:
        print("No factors to analyze")
        return

    returns = factors['rsi'].pct_change().dropna()

    ic = FactorAnalyzer.compute_ic(factors['rsi'].dropna(), returns)
    print(f"\nIC (Information Coefficient): {ic:.4f}")

    ir = FactorAnalyzer.compute_ir(factors['rsi'].dropna(), returns)
    print(f"IR (Information Ratio): {ir:.4f}")

    corr_matrix = FactorAnalyzer.factor_correlation(factors[['rsi', 'atr', 'momentum', 'volume_ratio']])
    print("\nFactor Correlation Matrix:")
    print(corr_matrix.round(3))

    print("\n" + "=" * 60)
    print("Factor Analysis Test PASSED")
    print("=" * 60)


def main():
    print("\n" + "=" * 60)
    print("MyQuant Factor Library Test Suite")
    print("=" * 60)

    factors = test_technical_factors()
    test_factor_analysis(factors)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
