"""
MyQuant 数据源诊断脚本 - 临时版本
"""

import os
import sys
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import requests

DIAGNOSIS_RESULTS = []


def log_step(
    name: str,
    status: str,
    elapsed: float,
    error_type: str = None,
    error_msg: str = None,
    details: str = None
) -> None:
    result = {
        'step': name,
        'status': status,
        'elapsed_seconds': elapsed,
        'error_type': error_type,
        'error_message': error_msg,
        'details': details
    }
    DIAGNOSIS_RESULTS.append(result)
    
    print(f"\n[{status}] {name}")
    print(f"   Elapsed: {elapsed:.2f}s")
    if error_type:
        print(f"   Error: {error_type}")
    if error_msg:
        print(f"   Message: {error_msg[:200]}")
    if details:
        print(f"   Details: {details}")


def print_separator(title: str) -> None:
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def step_1_environment_info() -> None:
    print_separator("Step 1: Environment & Version Info")
    start_time = time.time()
    
    try:
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        print(f"  Python version: {python_version}")
        
        print(f"  pandas: {pd.__version__}")
        
        try:
            import akshare
            akshare_version = akshare.__version__
            print(f"  akshare: {akshare_version}")
        except ImportError:
            akshare_version = 'NOT_INSTALLED'
            print(f"  akshare: NOT INSTALLED")
        
        print(f"  requests: {requests.__version__}")
        
        proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY']
        for var in proxy_vars:
            val = os.environ.get(var)
            print(f"  {var}: {val}")
        
        elapsed = time.time() - start_time
        log_step(
            'Environment Info', 'OK', elapsed,
            details=f"Python={python_version}, akshare={akshare_version}"
        )
        
    except Exception as e:
        elapsed = time.time() - start_time
        log_step(
            'Environment Info', 'FAIL', elapsed,
            error_type=type(e).__name__,
            error_msg=str(e)
        )


def step_2_requests_direct_connect() -> None:
    print_separator("Step 2: Direct Requests Connection (trust_env=False)")
    start_time = time.time()
    
    session = requests.Session()
    session.trust_env = False
    
    try:
        url = (
            'https://push2his.eastmoney.com/api/qt/stock/kline/get?'
            'fields1=f1,f2,f3,f4,f5,f6&'
            'fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116&'
            'ut=7eea3edcaed734bea9cbfc24409ed989&'
            'klt=101&fqt=1&secid=0.000001&'
            'beg=20250101&end=20250131'
        )
        response = session.get(url, timeout=15)
        
        print(f"  Eastmoney API: status={response.status_code}")
        print(f"  Response len: {len(response.content)}")
        
        klines = []
        if response.status_code == 200:
            data = response.json()
            print(f"  Response code: {data.get('rc')}")
            print(f"  Has data: {bool(data.get('data'))}")
            
            if data.get('data'):
                klines = data['data'].get('klines', [])
                print(f"  Kline count: {len(klines)}")
                if klines:
                    print(f"  First kline: {klines[0][:50]}")
            
            elapsed = time.time() - start_time
            log_step(
                'Direct Requests', 'OK', elapsed,
                details=f"status={response.status_code}, klines={len(klines)}"
            )
        else:
            elapsed = time.time() - start_time
            log_step(
                'Direct Requests', 'WARN', elapsed,
                details=f"status={response.status_code}"
            )
            
    except Exception as e:
        elapsed = time.time() - start_time
        log_step(
            'Direct Requests', 'FAIL', elapsed,
            error_type=type(e).__name__,
            error_msg=str(e)
        )


def step_3_akshare_single_stock() -> None:
    print_separator("Step 4: AkShare Single Stock")
    start_time = time.time()
    
    try:
        import akshare as ak
        
        import requests
        original_get = requests.get
        
        direct_session = requests.Session()
        direct_session.trust_env = False
        
        def patched_get(url, **kwargs):
            kwargs.pop('proxies', None)
            return direct_session.get(url, **kwargs)
        
        requests.get = patched_get
        
        try:
            df = ak.stock_zh_a_hist(
                symbol='000001',
                start_date='20250101',
                end_date='20250131',
                adjust='qfq'
            )
            
            print(f"  AkShare result shape: {df.shape}")
            print(f"  AkShare columns: {list(df.columns)}")
            
            elapsed = time.time() - start_time
            log_step(
                'AkShare Single Stock', 'OK', elapsed,
                details=f"rows={len(df)}, columns={len(df.columns)}"
            )
        finally:
            requests.get = original_get
            
    except ImportError:
        elapsed = time.time() - start_time
        log_step(
            'AkShare Single Stock', 'FAIL', elapsed,
            error_type='ImportError',
            error_msg='akshare not installed'
        )
    except Exception as e:
        elapsed = time.time() - start_time
        log_step(
            'AkShare Single Stock', 'FAIL', elapsed,
            error_type=type(e).__name__,
            error_msg=str(e)
        )


def generate_report() -> None:
    print_separator("Final Report")
    
    ok_count = sum(1 for r in DIAGNOSIS_RESULTS if r['status'] == 'OK')
    warn_count = sum(1 for r in DIAGNOSIS_RESULTS if r['status'] == 'WARN')
    fail_count = sum(1 for r in DIAGNOSIS_RESULTS if r['status'] == 'FAIL')
    
    print(f"\n  Summary:")
    print(f"    OK: {ok_count}")
    print(f"    WARN: {warn_count}")
    print(f"    FAIL: {fail_count}")
    
    os.makedirs('reports', exist_ok=True)
    
    report_path = 'reports/data_source_diagnosis_report.md'
    
    report_lines = []
    report_lines.append("# Data Source Diagnosis Report")
    report_lines.append("")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append("## Summary")
    report_lines.append("")
    report_lines.append("| Status | Count |")
    report_lines.append("|--------|-------|")
    report_lines.append(f"| OK | {ok_count} |")
    report_lines.append(f"| WARN | {warn_count} |")
    report_lines.append(f"| FAIL | {fail_count} |")
    report_lines.append("")
    report_lines.append("## Detailed Steps")
    report_lines.append("")
    
    for result in DIAGNOSIS_RESULTS:
        err_type = result.get('error_type', '') or ''
        err_msg = (result.get('error_message', '') or '')[:50].replace('\n', ' ')
        details = (result.get('details', '') or '')[:100].replace('\n', ' ')
        
        report_lines.append(f"### {result['step']}")
        report_lines.append(f"- Status: {result['status']}")
        report_lines.append(f"- Elapsed: {result['elapsed_seconds']:.2f}s")
        if err_type:
            report_lines.append(f"- Error Type: {err_type}")
        if err_msg:
            report_lines.append(f"- Error: {err_msg}")
        if details:
            report_lines.append(f"- Details: {details}")
        report_lines.append("")
    
    report_content = "\n".join(report_lines)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n  Report saved to: {report_path}")


def main():
    print("="*80)
    print("  MyQuant Data Source Diagnosis")
    print("="*80)
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    step_1_environment_info()
    step_2_requests_direct_connect()
    step_3_akshare_single_stock()
    
    generate_report()
    
    print("\n" + "="*80)
    print("  Diagnosis Complete!")
    print("="*80)


if __name__ == '__main__':
    main()
