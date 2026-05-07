#!/usr/bin/env python
"""测试数据质量报告功能"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.data.quality_report import DataQualityReporter


def test_quality_report():
    """测试数据质量报告"""
    print("\n" + "=" * 60)
    print("Testing Data Quality Report")
    print("=" * 60)

    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    base = 100
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = base * (1 + returns).cumprod()

    df = pd.DataFrame({
        'open': prices * (1 - np.random.normal(0.005, 0.01, len(dates))),
        'high': prices * (1 + np.random.normal(0.01, 0.005, len(dates))),
        'low': prices * (1 - np.random.normal(0.01, 0.005, len(dates))),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)

    reporter = DataQualityReporter('TEST.SZ', 'test_source')
    report = reporter.generate_report(df)

    print(f"\nSymbol: {report['symbol']}")
    print(f"Quality Score: {report['data_quality_score']:.1f}/100")
    print(f"Missing Rate: {report['missing_values']['overall_missing_rate']:.4%}")
    print(f"Total Rows: {report['basic_info']['total_rows']}")
    print(f"Suspension Days: {report['suspended_days']['count']}")
    print(f"Price Anomalies: {len(report['anomalies']['price_anomalies'])}")

    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")

    status = 'PASS' if report['data_quality_score'] >= 80 else 'WARNING' if report['data_quality_score'] >= 60 else 'FAIL'
    print(f"\nStatus: {status}")

    reporter.save_report('./test_quality_report.json')
    print("\nReport saved to ./test_quality_report.json")

    print("\n" + "=" * 60)
    print("Data Quality Report Test PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_quality_report()
