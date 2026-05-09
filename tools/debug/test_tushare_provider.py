"""
Tushare Provider Smoke Test

安全要求:
1. 不打印 TUSHARE_TOKEN
2. 不记录 TUSHARE_TOKEN
3. 只从环境变量读取
4. 如果没有 token，返回 unavailable

测试内容:
1. 检查 TUSHARE_TOKEN 是否存在
2. 测试 stock_basic
3. 测试 trade_cal
4. 测试 daily
5. 测试 adj_factor
6. 测试 daily_basic
7. 测试 stk_limit
8. 测试 suspend_d
9. 测试多只股票日频数据获取
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd

from src.data.sources.tushare_provider import TushareProvider


def main():
    print("=" * 80)
    print("Tushare Provider Smoke Test")
    print("=" * 80)
    
    results = {
        'token_present': False,
        'token_printed': False,
        'interfaces_tested': [],
        'success_count': 0,
        'failed_count': 0,
        'failed_reasons': {},
        'standardized_fields': [],
        'can_use_for_research': False,
        'can_use_for_live_trading': False
    }
    
    # 1. 检查 token
    print("\n1. Checking TUSHARE_TOKEN...")
    token = os.environ.get('TUSHARE_TOKEN')
    if token:
        print("   [OK] TUSHARE_TOKEN present")
        print("   [OK] Token not printed for security")
        results['token_present'] = True
    else:
        print("   [FAIL] TUSHARE_TOKEN not found in environment")
        results['failed_reasons']['token'] = "TUSHARE_TOKEN environment variable not set"
        results['failed_count'] += 1
        
        # 如果没有 token，直接输出报告
        generate_report(results)
        return
    
    # 2. 初始化 provider
    print("\n2. Initializing TushareProvider...")
    provider = TushareProvider()
    
    if not provider.is_available():
        print("   [FAIL] TushareProvider not available")
        results['failed_reasons']['init'] = "TushareProvider initialization failed"
        results['failed_count'] += 1
        generate_report(results)
        return
    
    print("   [OK] TushareProvider initialized successfully")
    
    # 3. 测试 stock_basic
    print("\n3. Testing stock_basic...")
    results['interfaces_tested'].append('stock_basic')
    stock_df, error = provider.get_stock_basic()
    if stock_df is not None and not stock_df.empty:
        print(f"   [OK] Got {len(stock_df)} stocks")
        print(f"   [OK] Columns: {list(stock_df.columns)}")
        results['success_count'] += 1
        results['standardized_fields'].extend(['stock', 'stock_name', 'industry', 'list_date', 'market'])
    else:
        print(f"   [FAIL] {error}")
        results['failed_reasons']['stock_basic'] = error
        results['failed_count'] += 1
    
    # 4. 测试 trade_cal
    print("\n4. Testing trade_cal...")
    results['interfaces_tested'].append('trade_cal')
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
    
    cal_df, error = provider.get_trade_cal(start_date, end_date)
    if cal_df is not None and not cal_df.empty:
        print(f"   [OK] Got {len(cal_df)} trading days")
        results['success_count'] += 1
        results['standardized_fields'].extend(['date', 'is_trading_day'])
    else:
        print(f"   [WARN] {error}")
        results['failed_reasons']['trade_cal'] = error
        results['failed_count'] += 1
    
    # 获取测试股票列表
    test_stocks = []
    if stock_df is not None and not stock_df.empty:
        test_stocks = stock_df['stock'].head(10).tolist()
    else:
        test_stocks = ['000001.SZ', '600000.SH', '000002.SZ', '600519.SH']
    
    print(f"\n5. Testing daily data for {len(test_stocks)} stocks...")
    results['interfaces_tested'].append('daily')
    
    daily_success = 0
    daily_failed = 0
    
    for stock in test_stocks[:3]:  # 只测试前3只
        time.sleep(0.05)  # 限速
        df, error = provider.get_daily(stock, start_date, end_date)
        if df is not None and not df.empty:
            daily_success += 1
        else:
            daily_failed += 1
            if stock not in results['failed_reasons']:
                results['failed_reasons'][stock] = error or "No data"
    
    if daily_success > 0:
        print(f"   [OK] Daily data: {daily_success} success, {daily_failed} failed")
        results['success_count'] += 1
        results['standardized_fields'].extend(['date', 'stock', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_change'])
    else:
        print(f"   [FAIL] Daily data all failed")
        results['failed_reasons']['daily'] = "All daily requests failed"
        results['failed_count'] += 1
    
    # 6. 测试 adj_factor
    print("\n6. Testing adj_factor...")
    results['interfaces_tested'].append('adj_factor')
    
    adj_success = 0
    for stock in test_stocks[:2]:
        time.sleep(0.05)
        df, error = provider.get_adj_factor(stock, start_date, end_date)
        if df is not None and not df.empty:
            adj_success += 1
    
    if adj_success > 0:
        print(f"   [OK] adj_factor: {adj_success} success")
        results['success_count'] += 1
        results['standardized_fields'].append('adj_factor')
    else:
        print(f"   [WARN] adj_factor failed (may require higher permission)")
        results['failed_reasons']['adj_factor'] = "Permission denied or no data"
    
    # 7. 测试 daily_basic
    print("\n7. Testing daily_basic...")
    results['interfaces_tested'].append('daily_basic')
    
    basic_success = 0
    for stock in test_stocks[:2]:
        time.sleep(0.05)
        df, error = provider.get_daily_basic(stock, start_date, end_date)
        if df is not None and not df.empty:
            basic_success += 1
    
    if basic_success > 0:
        print(f"   [OK] daily_basic: {basic_success} success")
        results['success_count'] += 1
        results['standardized_fields'].extend(['turnover', 'total_mv', 'circ_mv'])
    else:
        print(f"   [WARN] daily_basic failed (may require higher permission)")
        results['failed_reasons']['daily_basic'] = "Permission denied or no data"
    
    # 8. 测试 stk_limit
    print("\n8. Testing stk_limit...")
    results['interfaces_tested'].append('stk_limit')
    
    limit_success = 0
    for stock in test_stocks[:2]:
        time.sleep(0.05)
        df, error = provider.get_stk_limit(stock, start_date, end_date)
        if df is not None and not df.empty:
            limit_success += 1
    
    if limit_success > 0:
        print(f"   [OK] stk_limit: {limit_success} success")
        results['success_count'] += 1
        results['standardized_fields'].extend(['up_limit', 'down_limit'])
    else:
        print(f"   [WARN] stk_limit failed (may require higher permission)")
        results['failed_reasons']['stk_limit'] = "Permission denied or no data"
    
    # 9. 测试 suspend_d
    print("\n9. Testing suspend_d...")
    results['interfaces_tested'].append('suspend_d')
    
    suspend_success = 0
    for stock in test_stocks[:2]:
        time.sleep(0.05)
        df, error = provider.get_suspend_d(stock, start_date, end_date)
        if df is not None:
            suspend_success += 1
    
    if suspend_success > 0:
        print(f"   [OK] suspend_d: {suspend_success} success")
        results['success_count'] += 1
        results['standardized_fields'].append('is_suspended')
    else:
        print(f"   [WARN] suspend_d failed (may require higher permission)")
        results['failed_reasons']['suspend_d'] = "Permission denied or no data"
    
    # 10. 综合测试：获取 10 只股票 3 个月数据
    print("\n10. Testing fetch_daily_history (10 stocks, 3 months)...")
    three_months_ago = (datetime.now() - timedelta(days=90)).strftime('%Y%m%d')
    
    history_success = 0
    history_failed = 0
    all_data = []
    
    for i, stock in enumerate(test_stocks[:10]):
        time.sleep(0.02)  # Tushare 限速
        df, error = provider.fetch_daily_history(stock, three_months_ago, end_date)
        
        if df is not None and not df.empty:
            history_success += 1
            all_data.append(df)
            if i < 3:
                print(f"      {stock}: {len(df)} rows")
        else:
            history_failed += 1
    
    print(f"   [Result] {history_success} success, {history_failed} failed")
    
    if history_success >= 5:
        print("   [OK] Can fetch sufficient data for research")
        results['can_use_for_research'] = True
    else:
        print("   [WARN] May not have enough data for research")
    
    # 保存测试数据
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        output_dir = 'data/processed/profiles/tushare_smoke_test'
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存数据
        price_path = os.path.join(output_dir, 'prices.parquet')
        combined_df.to_parquet(price_path, index=False)
        print(f"\n   [OK] Test data saved to {price_path}")
        
        # 保存 metadata
        metadata = {
            'profile': 'tushare_smoke_test',
            'data_source_used': 'tushare',
            'target_stock_count': 10,
            'actual_stock_count': history_success,
            'test_date': datetime.now().isoformat(),
            'success_symbols': [s for i, s in enumerate(test_stocks[:10]) if i < history_success],
            'failed_symbols': [s for i, s in enumerate(test_stocks[:10]) if i >= history_success],
            'can_use_for_research': results['can_use_for_research'],
            'can_use_for_live_trading': False
        }
        metadata_path = os.path.join(output_dir, 'prices_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    # 生成报告
    generate_report(results)


def generate_report(results):
    """生成测试报告"""
    print("\n" + "=" * 80)
    print("Tushare Smoke Test Report")
    print("=" * 80)
    
    report_lines = []
    report_lines.append("# Tushare Provider Smoke Test Report")
    report_lines.append("")
    report_lines.append("## Test Summary")
    report_lines.append("")
    report_lines.append("| Item | Value |")
    report_lines.append("|------|-------|")
    report_lines.append(f"| TUSHARE_TOKEN Present | {results['token_present']} |")
    report_lines.append(f"| Token Printed | {results['token_printed']} |")
    report_lines.append(f"| Interfaces Tested | {len(results['interfaces_tested'])} |")
    report_lines.append(f"| Success Count | {results['success_count']} |")
    report_lines.append(f"| Failed Count | {results['failed_count']} |")
    report_lines.append(f"| Can Use For Research | {results['can_use_for_research']} |")
    report_lines.append(f"| Can Use For Live Trading | {results['can_use_for_live_trading']} |")
    
    report_lines.append("")
    report_lines.append("## Interfaces Tested")
    report_lines.append("")
    for interface in results['interfaces_tested']:
        report_lines.append(f"- {interface}")
    
    report_lines.append("")
    report_lines.append("## Standardized Fields")
    report_lines.append("")
    unique_fields = sorted(list(set(results['standardized_fields'])))
    for field in unique_fields:
        report_lines.append(f"- {field}")
    
    if results['failed_reasons']:
        report_lines.append("")
        report_lines.append("## Failed Reasons")
        report_lines.append("")
        for key, reason in results['failed_reasons'].items():
            report_lines.append(f"- {key}: {reason}")
    
    # 保存报告
    os.makedirs('reports', exist_ok=True)
    report_path = os.path.join('reports', 'tushare_smoke_test_report.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\nReport saved to: {report_path}")
    
    # 打印摘要
    print("\n" + "=" * 80)
    if results['can_use_for_research']:
        print("✅ TEST PASSED: Tushare is ready for research use")
    else:
        print("⚠️ TEST WARN: Tushare may not have sufficient data")
    print("=" * 80)


if __name__ == '__main__':
    main()
