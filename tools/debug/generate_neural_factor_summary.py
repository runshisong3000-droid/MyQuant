"""
生成 neural_factor_summary.parquet

从 neural_factors.parquet 提取因子并评估，生成包含 RankIC、ICIR、coverage 等指标的 summary。
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.factors.auto.factor_evaluator import FactorEvaluator


def main():
    profile = 'research_medium_trial'
    dashboard_dir = f'data/dashboard/profiles/{profile}'
    
    # 加载 neural_factors
    neural_factors_path = os.path.join(dashboard_dir, 'neural_factors.parquet')
    if not os.path.exists(neural_factors_path):
        print(f"ERROR: {neural_factors_path} 不存在")
        return 1
    
    df = pd.read_parquet(neural_factors_path)
    print(f"Loaded neural_factors: {df.shape}")
    
    # 获取因子列
    factor_columns = [col for col in df.columns if col.startswith('neural_factor_')]
    print(f"Factor columns: {factor_columns}")
    
    # 需要加载 future_return 来计算 IC
    processed_dir = f'data/processed/profiles/{profile}'
    prices_path = os.path.join(processed_dir, 'prices.parquet')
    
    if not os.path.exists(prices_path):
        print(f"ERROR: {prices_path} 不存在")
        return 1
    
    prices_df = pd.read_parquet(prices_path)
    
    # 计算 future_return（按股票分组）
    prices_df = prices_df.sort_values(['stock', 'date'])
    prices_df['future_return_1d'] = prices_df.groupby('stock')['close_raw'].pct_change().shift(-1)
    
    # 创建 MultiIndex
    df['signal_date'] = pd.to_datetime(df['signal_date'])
    df = df.set_index(['signal_date', 'stock'])
    
    prices_df['date'] = pd.to_datetime(prices_df['date'])
    prices_df = prices_df.set_index(['date', 'stock'])
    
    # 对齐索引
    future_return = prices_df['future_return_1d']
    
    # 评估每个因子
    evaluator = FactorEvaluator()
    results = []
    
    for factor_name in factor_columns:
        factor_series = df[factor_name]
        
        # 计算 RankIC
        common_index = factor_series.index.intersection(future_return.index)
        if len(common_index) == 0:
            print(f"WARNING: No common index for {factor_name}")
            continue
        
        factor_vals = factor_series.loc[common_index]
        return_vals = future_return.loc[common_index]
        
        # 去除 NaN
        valid_mask = ~factor_vals.isna() & ~return_vals.isna()
        factor_vals = factor_vals[valid_mask]
        return_vals = return_vals[valid_mask]
        
        if len(factor_vals) == 0:
            print(f"WARNING: No valid data for {factor_name}")
            continue
        
        # 计算每日 RankIC
        daily_ic = []
        dates = factor_vals.index.get_level_values('signal_date').unique()
        
        for date in dates:
            date_mask = factor_vals.index.get_level_values('signal_date') == date
            f_vals = factor_vals[date_mask]
            r_vals = return_vals[date_mask]
            
            if len(f_vals) >= 10:  # 至少10只股票
                ic = np.corrcoef(f_vals.rank(), r_vals)[0, 1]
                if not np.isnan(ic):
                    daily_ic.append(ic)
        
        if daily_ic:
            rank_ic_mean = np.mean(daily_ic)
            ic_std = np.std(daily_ic)
            icir = rank_ic_mean / ic_std if ic_std > 0 else 0
        else:
            rank_ic_mean = 0
            icir = 0
        
        coverage = len(factor_vals) / len(factor_series)
        
        results.append({
            'factor_name': factor_name,
            'rank_ic': rank_ic_mean,
            'icir': icir,
            'coverage': coverage,
            'profile': profile,
            'data_source': 'tushare',
            'can_use_for_live_trading': False
        })
        
        print(f"Evaluated {factor_name}: RankIC={rank_ic_mean:.4f}, ICIR={icir:.4f}, coverage={coverage:.4f}")
    
    # 创建 summary DataFrame
    summary_df = pd.DataFrame(results)
    
    # 保存
    output_path = os.path.join(dashboard_dir, 'neural_factor_summary.parquet')
    summary_df.to_parquet(output_path)
    print(f"\nSaved neural_factor_summary.parquet to: {output_path}")
    print(f"Summary shape: {summary_df.shape}")
    
    return 0


if __name__ == '__main__':
    exit(main())
