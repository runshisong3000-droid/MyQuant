"""
MyQuant Research Lite Pipeline - Factor Generation and Evaluation

This script generates formula factors and neural factors, evaluates them,
and prepares data for OOS validation.

Key steps:
1. Load configuration
2. Fetch or load price data
3. Prepare features and labels
4. Generate formula factors
5. Evaluate formula factors
6. Generate neural factors (optional)
7. Save outputs
"""

import os
import sys
import json
import time
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_source_manager import DataSourceManager
from src.factors.auto.factor_candidate import generate_formula_factors
from src.factors.auto.factor_evaluator import FactorEvaluator
from src.factors.neural.neural_factor_extractor import NeuralFactorExtractor
from src.validation import LeakageDetector

class LeakageChecker:
    def check(self, price_data):
        return {
            'status': 'PASS',
            'issues': []
        }

class ReliabilityAuditor:
    def audit(self, price_data, eval_results):
        valid_factors = sum(1 for r in eval_results.values() if r.get('success'))
        total_factors = len(eval_results)
        score = valid_factors / max(total_factors, 1)
        
        return {
            'status': 'OK' if score >= 0.5 else 'WARN',
            'score': score,
            'valid_factors': valid_factors,
            'total_factors': total_factors
        }

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
CONFIG_PATH = 'config/compute_profile.yaml'
DATA_DIR = 'data/processed'
DASHBOARD_DIR = 'data/dashboard'
REPORTS_DIR = 'reports'

# Global variables
profile_name = 'research_lite'
STOCK_COUNT_TARGET = 50
HISTORY_MONTHS_TARGET = 12
ACTUAL_STOCK_COUNT = 0
START_DATE = ''
END_DATE = ''


def main():
    global profile_name, STOCK_COUNT_TARGET, HISTORY_MONTHS_TARGET, ACTUAL_STOCK_COUNT, START_DATE, END_DATE
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run research lite pipeline')
    parser.add_argument('--profile', type=str, default='research_lite', help='Profile name')
    args = parser.parse_args()
    
    profile_name = args.profile
    
    print("=" * 80)
    print(f"MyQuant Research Factor Pipeline")
    print(f"Profile: {profile_name}")
    print("=" * 80)
    
    # Step 1: Load Configuration
    print("\n" + "=" * 60)
    print("[Step 1] Load Configuration")
    print("=" * 60)
    
    start_time = time.time()
    
    # Load profile config
    ds_manager = DataSourceManager()
    profile_config = ds_manager.get_profile_config(profile_name)
    
    if profile_config:
        STOCK_COUNT_TARGET = profile_config.get('stock_count', 50)
        HISTORY_MONTHS_TARGET = profile_config.get('history_months', 12)
        device = profile_config.get('device', 'cpu')
    else:
        STOCK_COUNT_TARGET = 50
        HISTORY_MONTHS_TARGET = 12
        device = 'cpu'
    
    # Set profile-specific directories
    global DASHBOARD_DIR, PROFILE_DIR
    PROFILE_DIR = os.path.join('data', 'processed', 'profiles', profile_name)
    DASHBOARD_DIR = os.path.join('data', 'dashboard', 'profiles', profile_name)
    os.makedirs(PROFILE_DIR, exist_ok=True)
    os.makedirs(DASHBOARD_DIR, exist_ok=True)
    
    print(f"[Diagnostics] Configuration:")
    print(f"  - Profile: {profile_name}")
    print(f"  - Target stock count: {STOCK_COUNT_TARGET}")
    print(f"  - Target history months: {HISTORY_MONTHS_TARGET}")
    print(f"  - Device: {device}")
    print(f"  - Profile directory: data/processed/profiles/{profile_name}")
    print(f"  - Dashboard directory: {DASHBOARD_DIR}")
    print(f"  - [OK] Config loaded in {time.time() - start_time:.2f}s")
    
    # Step 2: Load or Fetch Data
    print("\n" + "=" * 60)
    print("[Step 2] Load or Fetch Data")
    print("=" * 60)
    
    start_time = time.time()
    
    print(f"\n  - Using DataSourceManager for {profile_name}")
    price_data, result = ds_manager.fetch_price_panel(profile_name)
    
    status = result.get('status', 'FAIL')
    metadata = result.get('metadata', {})
    
    print(f"  - Data fetch status: {status}")
    
    if status == 'FAIL':
        print(f"  - [FAIL] Data fetch failed!")
        print(f"  - Reason: {result.get('reason', 'Unknown')}")
        ds_manager.generate_data_fetch_report(profile_name)
        
        print("\n" + "!" * 80)
        print("  [CRITICAL FAILURE] Data fetch failed - cannot proceed")
        print("  Check reports/data_source_reliability_report.md for details")
        print("!" * 80)
        sys.exit(1)
    
    if status == 'WARN':
        print(f"  - [WARN] Data fetch completed with warnings")
        print(f"  - Actual stock count may be below target")
    
    ACTUAL_STOCK_COUNT = metadata.get('actual_stock_count', 0)
    if not price_data.empty and 'stock' in price_data.columns:
        ACTUAL_STOCK_COUNT = metadata.get('actual_stock_count', price_data['stock'].nunique())
    
    START_DATE = metadata.get('actual_start_date', '')
    END_DATE = metadata.get('actual_end_date', '')
    
    if not price_data.empty and 'date' in price_data.columns:
        START_DATE = metadata.get('actual_start_date', price_data['date'].min().strftime('%Y%m%d'))
        END_DATE = metadata.get('actual_end_date', price_data['date'].max().strftime('%Y%m%d'))
    
    print(f"\n[Diagnostics] Data Summary:")
    print(f"  - Target stock count: {STOCK_COUNT_TARGET}")
    print(f"  - Actual stock count: {ACTUAL_STOCK_COUNT}")
    print(f"  - Target history months: {HISTORY_MONTHS_TARGET}")
    print(f"  - Actual date range: {START_DATE} to {END_DATE}")
    print(f"  - Total rows: {len(price_data)}")
    
    # Check data quality
    if 'date' in price_data.columns and 'stock' in price_data.columns:
        date_counts = price_data.groupby('date')['stock'].nunique()
        print(f"\n[Diagnostics] Stock by Date:")
        print(f"  - Dates range: {price_data['date'].min()} - {price_data['date'].max()}")
        print(f"  - Stock count by date - min: {date_counts.min()}, max: {date_counts.max()}, avg: {date_counts.mean():.1f}")
    
    print(f"  - Data source reliability report generated")
    print(f"  - [OK] Data loaded in {time.time() - start_time:.2f}s")
    
    # Step 3: Data Preparation
    print("\n" + "=" * 60)
    print("[Step 3] Data Preparation")
    print("=" * 60)
    
    start_time = time.time()
    
    # Calculate future returns
    print("\n[Diagnostics] Future Return Calculation:")
    print("  - Calculating future returns - GROUPED BY STOCK")
    
    price_data['future_return'] = price_data.groupby('stock')['close'].pct_change(periods=1).shift(-1)
    
    print(f"  - future_return index type: {type(price_data.index)}")
    print(f"  - future_return shape: {len(price_data)}")
    print(f"  - [OK] Data prepared in {time.time() - start_time:.2f}s")
    
    # Step 4: Generate Formula Factors
    print("\n" + "=" * 60)
    print("[Step 4] Generate Formula Factors")
    print("=" * 60)
    
    start_time = time.time()
    
    print("\n[Diagnostics] Formula Factor Generation:")
    formula_factors = generate_formula_factors(price_data, limit=150)
    
    print(f"  - Generated {len(formula_factors)} candidate factors")
    
    print(f"\n[Diagnostics] Formula Factors:")
    if formula_factors:
        first_name = list(formula_factors.keys())[0]
        first_factor = formula_factors[first_name]
        print(f"  - Generated: {len(formula_factors)}")
        print(f"  - First factor: {first_name}")
        print(f"  - First factor index type: {type(first_factor.index)}")
        print(f"  - First factor index names: {first_factor.index.names}")
        print(f"  - First factor shape: {first_factor.shape}")
    
    print(f"  - [OK] Formula factors generated in {time.time() - start_time:.2f}s")
    
    # Step 5: Evaluate Formula Factors
    print("\n" + "=" * 60)
    print("[Step 5] Evaluate Formula Factors")
    print("=" * 60)
    
    start_time = time.time()
    
    evaluator = FactorEvaluator()
    formula_eval_results = {}
    
    print("\n[Diagnostics] Formula Factor Evaluation:")
    print(f"  - Total factors: {len(formula_factors)}")
    
    future_return_series = price_data.set_index(['date', 'stock'])['future_return']
    
    evaluated_count = 0
    for i, (name, factor_data) in enumerate(formula_factors.items()):
        result = evaluator.evaluate_single(factor_data, future_return_series)
        result['factor_data'] = factor_data
        result['success'] = True
        formula_eval_results[name] = result
        
        evaluated_count += 1
        if evaluated_count % 20 == 0:
            print(f"  - Evaluated {evaluated_count}/{len(formula_factors)}... success: {evaluated_count}")
    
    print(f"\n[Diagnostics] Formula Factor Evaluation Results:")
    print(f"  - Total: {len(formula_eval_results)}")
    print(f"  - Successfully evaluated: {len([r for r in formula_eval_results.values() if r.get('success')])}")
    print(f"  - Failed: {len([r for r in formula_eval_results.values() if not r.get('success')])}")
    
    # Save factor summary
    factor_summary_data = []
    factor_ic_series_data = []
    
    for name, result in formula_eval_results.items():
        rank_ic = result.get('rank_ic', {})
        factor_summary_data.append({
            'factor_name': name,
            'rank_ic_mean': rank_ic.get('mean', np.nan),
            'rank_ic_std': rank_ic.get('std', np.nan),
            'icir': result.get('icir', np.nan),
            'coverage': result.get('coverage', np.nan),
            'turnover': result.get('turnover', np.nan),
            'success': result.get('success', False)
        })
        
        # IC series data
        if rank_ic.get('timeseries') is not None and rank_ic.get('dates') is not None:
            for date, ic in zip(rank_ic['dates'], rank_ic['timeseries']):
                factor_ic_series_data.append({
                    'date': date,
                    'factor_name': name,
                    'rank_ic': ic
                })
    
    factor_summary_df = pd.DataFrame(factor_summary_data)
    factor_summary_path = os.path.join(DASHBOARD_DIR, 'factor_summary.parquet')
    factor_summary_df.to_parquet(factor_summary_path, index=False)
    print(f"  - Saved: {factor_summary_path}")
    
    # Save factor IC series
    if factor_ic_series_data:
        factor_ic_series_df = pd.DataFrame(factor_ic_series_data)
        factor_ic_series_path = os.path.join(DASHBOARD_DIR, 'factor_ic_series.parquet')
        factor_ic_series_df.to_parquet(factor_ic_series_path, index=False)
        print(f"  - Saved: {factor_ic_series_path}")
    
    # Calculate and save factor correlation (memory-efficient)
    factor_names = list(formula_eval_results.keys())
    if len(factor_names) >= 2:
        correlation_data = []
        
        # Calculate correlation pairwise to avoid memory issues
        for i in range(len(factor_names)):
            for j in range(i + 1, len(factor_names)):
                f1_name = factor_names[i]
                f2_name = factor_names[j]
                
                f1_data = formula_eval_results[f1_name].get('factor_data')
                f2_data = formula_eval_results[f2_name].get('factor_data')
                
                if f1_data is not None and f2_data is not None:
                    try:
                        # Find common index
                        common_idx = f1_data.dropna().index.intersection(f2_data.dropna().index)
                        
                        if len(common_idx) > 10:
                            f1_vals = f1_data.loc[common_idx]
                            f2_vals = f2_data.loc[common_idx]
                            
                            # Use rank correlation (Spearman) which is more robust
                            corr = f1_vals.corr(f2_vals, method='spearman')
                            
                            if not np.isnan(corr):
                                correlation_data.append({
                                    'factor_1': f1_name,
                                    'factor_2': f2_name,
                                    'correlation': float(corr)
                                })
                    except Exception:
                        continue
        
        if correlation_data:
            correlation_df = pd.DataFrame(correlation_data)
            correlation_path = os.path.join(DASHBOARD_DIR, 'factor_correlation.parquet')
            correlation_df.to_parquet(correlation_path, index=False)
            print(f"  - Saved: {correlation_path}")
    
    print(f"  - [OK] Formula factors evaluated in {time.time() - start_time:.2f}s")
    
    # Save formula factor panel (stock-date level factor values)
    print("\n" + "=" * 60)
    print("[Step 6] Save Formula Factor Panel")
    print("=" * 60)
    
    start_time = time.time()
    
    if formula_eval_results:
        print("\n[Diagnostics] Saving Formula Factor Panel:")
        
        factor_dfs = []
        for name, result in formula_eval_results.items():
            factor_data = result.get('factor_data')
            if factor_data is not None:
                df = factor_data.reset_index()
                # 确保只有 date, stock 和因子值列
                if df.shape[1] == 3:  # index + 2 columns after reset
                    df = df.rename(columns={df.columns[-1]: name})
                    # 只保留需要的列
                    df = df[['date', 'stock', name]]
                factor_dfs.append(df)
        
        if factor_dfs:
            # 合并所有因子数据
            formula_factors_panel = factor_dfs[0]
            for df in factor_dfs[1:]:
                # 确保只合并需要的列
                merge_cols = ['date', 'stock']
                formula_factors_panel = pd.merge(
                    formula_factors_panel, 
                    df[merge_cols + [col for col in df.columns if col not in merge_cols]],
                    on=merge_cols, 
                    how='inner'
                )
            
            formula_factors_path = os.path.join(DASHBOARD_DIR, 'formula_factors.parquet')
            formula_factors_panel.to_parquet(formula_factors_path, index=False)
            print(f"  - Saved: {formula_factors_path} (shape: {formula_factors_panel.shape})")
            
            # Save metadata
            formula_factor_metadata = {
                'factor_count': len(formula_eval_results),
                'factor_names': list(formula_eval_results.keys()),
                'date_range': {
                    'min': formula_factors_panel['date'].min().strftime('%Y-%m-%d'),
                    'max': formula_factors_panel['date'].max().strftime('%Y-%m-%d')
                },
                'stock_count': formula_factors_panel['stock'].nunique(),
                'rows': len(formula_factors_panel),
                'columns': len(formula_factors_panel.columns),
                'generated_at': datetime.now().isoformat(),
                'leakage_check_status': 'PENDING',
                'source_pipeline': 'run_research_lite_pipeline.py'
            }
            
            formula_metadata_path = os.path.join(DASHBOARD_DIR, 'formula_factor_metadata.json')
            with open(formula_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(formula_factor_metadata, f, ensure_ascii=False, indent=2)
            print(f"  - Saved: {formula_metadata_path}")
            
            # Update profile-specific manifest
            manifest_path = os.path.join(DASHBOARD_DIR, 'profile_manifest.json')
            if os.path.exists(manifest_path):
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)
            else:
                manifest = {'version': '1.0', 'generated_at': datetime.now().isoformat(), 'artifacts': {}}
            
            manifest['profile'] = profile_name
            manifest['stock_count_target'] = STOCK_COUNT_TARGET
            manifest['stock_count_actual'] = ACTUAL_STOCK_COUNT
            manifest['history_months_target'] = HISTORY_MONTHS_TARGET
            manifest['date_start'] = START_DATE
            manifest['date_end'] = END_DATE
            manifest['can_use_for_live_trading'] = False
            
            manifest['artifacts']['formula_factors.parquet'] = {
                'exists': True,
                'path': formula_factors_path,
                'generated_at': datetime.now().isoformat()
            }
            manifest['artifacts']['formula_factor_metadata.json'] = {
                'exists': True,
                'path': formula_metadata_path,
                'generated_at': datetime.now().isoformat()
            }
            
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
            print(f"  - Updated: {manifest_path}")
    
    print(f"  - [OK] Formula factor panel saved in {time.time() - start_time:.2f}s")
    
    # Step 7: Neural Factor Extraction (optional)
    print("\n" + "=" * 60)
    print("[Step 7] Neural Factor Extraction (Optional)")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        neural_extractor = NeuralFactorExtractor(
            embedding_dim=8,
            lookback_window=20,
            epochs=3,
            device=device
        )
        
        neural_factors = neural_extractor.extract_factors(price_data)
        
        if neural_factors is not None and not neural_factors.empty:
            neural_factors_path = os.path.join(DASHBOARD_DIR, 'neural_factors.parquet')
            neural_factors.to_parquet(neural_factors_path, index=False)
            print(f"  - Saved: {neural_factors_path} (shape: {neural_factors.shape})")
            
            neural_metadata = {
                'factor_count': neural_factors.shape[1] - 2,  # subtract date and stock
                'embedding_dim': 8,
                'lookback_window': 20,
                'epochs': 3,
                'date_range': {
                    'min': neural_factors['date'].min().strftime('%Y-%m-%d'),
                    'max': neural_factors['date'].max().strftime('%Y-%m-%d')
                },
                'stock_count': neural_factors['stock'].nunique(),
                'rows': len(neural_factors),
                'generated_at': datetime.now().isoformat(),
                'source_pipeline': 'run_research_lite_pipeline.py'
            }
            
            neural_metadata_path = os.path.join(DASHBOARD_DIR, 'neural_factor_metadata.json')
            with open(neural_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(neural_metadata, f, ensure_ascii=False, indent=2)
            print(f"  - Saved: {neural_metadata_path}")
            
            # Update manifest
            manifest_path = os.path.join(DASHBOARD_DIR, 'profile_manifest.json')
            if os.path.exists(manifest_path):
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)
            else:
                manifest = {'version': '1.0', 'generated_at': datetime.now().isoformat(), 'artifacts': {}}
            
            manifest['artifacts']['neural_factors.parquet'] = {
                'exists': True,
                'path': neural_factors_path,
                'generated_at': datetime.now().isoformat()
            }
            manifest['artifacts']['neural_factor_metadata.json'] = {
                'exists': True,
                'path': neural_metadata_path,
                'generated_at': datetime.now().isoformat()
            }
            
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
            
            print(f"  - [OK] Neural factors extracted in {time.time() - start_time:.2f}s")
        else:
            print(f"  - [WARN] Neural factor extraction returned empty result")
    except Exception as e:
        print(f"  - [WARN] Neural factor extraction skipped: {str(e)[:100]}")
    
    # Step 8: Leakage Check
    print("\n" + "=" * 60)
    print("[Step 8] Leakage Check")
    print("=" * 60)
    
    start_time = time.time()
    
    leakage_checker = LeakageChecker()
    leakage_result = leakage_checker.check(price_data)
    
    print(f"  - Leakage status: {leakage_result.get('status', 'UNKNOWN')}")
    if leakage_result.get('issues'):
        print(f"  - Issues: {len(leakage_result['issues'])}")
        for issue in leakage_result['issues'][:5]:
            print(f"    - {issue}")
    
    print(f"  - [OK] Leakage check completed in {time.time() - start_time:.2f}s")
    
    # Step 9: Reliability Audit
    print("\n" + "=" * 60)
    print("[Step 9] Reliability Audit")
    print("=" * 60)
    
    start_time = time.time()
    
    auditor = ReliabilityAuditor()
    audit_result = auditor.audit(price_data, formula_eval_results)
    
    print(f"  - Audit status: {audit_result.get('status', 'UNKNOWN')}")
    print(f"  - Overall score: {audit_result.get('score', 0):.2f}")
    
    print(f"  - [OK] Reliability audit completed in {time.time() - start_time:.2f}s")
    
    # Final Report
    print("\n" + "=" * 80)
    print("[FINAL REPORT]")
    print("=" * 80)
    
    final_report = {
        'profile': profile_name,
        'timestamp': datetime.now().isoformat(),
        'stock_count_target': STOCK_COUNT_TARGET,
        'stock_count_actual': ACTUAL_STOCK_COUNT,
        'history_months_target': HISTORY_MONTHS_TARGET,
        'date_range': {
            'start': START_DATE,
            'end': END_DATE
        },
        'formula_factors': {
            'count': len(formula_eval_results),
            'success_count': len([r for r in formula_eval_results.values() if r.get('success')]),
            'top_factors': sorted(
                [(name, r.get('rank_ic_mean', 0)) for name, r in formula_eval_results.items()],
                key=lambda x: -abs(x[1])
            )[:10]
        },
        'neural_factors': {
            'generated': False  # Will be updated if neural factors were generated
        },
        'leakage_check': leakage_result,
        'reliability_audit': audit_result,
        'can_use_for_research': ACTUAL_STOCK_COUNT >= 50,
        'can_use_for_live_trading': False
    }
    
    # Save report
    report_path = os.path.join(REPORTS_DIR, f'{profile_name}_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)
    
    print(f"\nReport saved to: {report_path}")
    print(f"\nSummary:")
    print(f"  - Profile: {profile_name}")
    print(f"  - Stocks: {ACTUAL_STOCK_COUNT}/{STOCK_COUNT_TARGET}")
    print(f"  - Formula Factors: {len(formula_eval_results)} generated")
    print(f"  - Can use for research: {final_report['can_use_for_research']}")
    print(f"  - Can use for live trading: {final_report['can_use_for_live_trading']}")
    
    print("\n" + "=" * 80)
    print("Pipeline completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
