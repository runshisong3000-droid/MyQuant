"""
快速测试 Eastmoney Direct Provider
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.data.sources.eastmoney_direct import EastmoneyDirectProvider

print("=== Testing Eastmoney Direct Provider ===")

# 初始化
provider = EastmoneyDirectProvider(timeout=15, retry_times=2, retry_sleep=0.5)

# 测试单个股票
print("\n--- Testing single stock (000001.SZ) ---")
df, error = provider.fetch_daily_history(
    symbol='000001.SZ',
    start_date='20250101',
    end_date='20250131',
    adjust='qfq'
)

if df is not None:
    print(f"[OK] Success! Got {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    if len(df) > 0:
        print("\nSample data:")
        print(df.head())
else:
    print(f"[FAIL] Failed: {error}")

# 验证数据
if df is not None:
    print("\n--- Validating data ---")
    is_valid, issues = provider.validate_daily_bars(df)
    if is_valid:
        print("[OK] Data is valid")
    else:
        print(f"[WARN] Data has issues: {issues}")
