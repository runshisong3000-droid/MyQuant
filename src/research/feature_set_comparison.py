"""
FeatureSetComparison - 特征集对比

功能:
    - 构造 formula_only、neural_only、formula_plus_neural 三类特征集
    - 对齐 MultiIndex(date, stock)
    - 计算样本外指标
    - 输出对比结果
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from src.factors.auto.factor_evaluator import FactorEvaluator


class FeatureSetComparison:
    """
    特征集对比类
    
    支持三类特征集:
    1. formula_only: 仅公式因子
    2. neural_only: 仅神经因子
    3. formula_plus_neural: 公式因子 + 神经因子
    """
    
    def __init__(self):
        self.evaluator = FactorEvaluator()
        self.feature_sets = {}
        self.comparison_results = {}
    
    def load_formula_factors(
        self,
        factor_panel_path: str = 'data/dashboard/formula_factors.parquet',
        factor_summary_path: str = 'data/dashboard/factor_summary.parquet'
    ) -> pd.DataFrame:
        """
        加载公式因子面板（stock-date级别因子值）
        
        Args:
            factor_panel_path: 公式因子面板文件路径（必须包含 date, stock 列和因子值列）
            factor_summary_path: 因子汇总文件路径（备用）
        
        Returns:
            公式因子DataFrame，包含 date, stock 和因子值列
        
        Raises:
            ValueError: 如果公式因子面板不存在
        """
        if os.path.exists(factor_panel_path):
            try:
                df = pd.read_parquet(factor_panel_path)
                # 验证必需列
                required_cols = ['date', 'stock']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"Formula factor panel missing required columns: {missing_cols}")
                
                # 获取因子列（排除 date, stock）
                factor_cols = [col for col in df.columns if col not in ['date', 'stock']]
                if len(factor_cols) == 0:
                    raise ValueError("Formula factor panel has no factor columns")
                
                df['date'] = pd.to_datetime(df['date'])
                return df
            except Exception as e:
                raise ValueError(f"Failed to load formula factors panel: {e}")
        else:
            raise ValueError(f"Formula factor panel not found at {factor_panel_path}. "
                           "Please run run_research_lite_pipeline.py first.")
    
    def load_neural_factors(
        self,
        neural_factors_path: str = 'data/dashboard/neural_factors.parquet'
    ) -> pd.DataFrame:
        """
        加载神经因子
        
        Args:
            neural_factors_path: 神经因子文件路径
        
        Returns:
            神经因子DataFrame，MultiIndex(date, stock)
        """
        try:
            df = pd.read_parquet(neural_factors_path)
            return df
        except Exception as e:
            raise ValueError(f"Failed to load neural factors: {e}")
    
    def load_future_return(
        self,
        prices_path: str = 'data/processed/research_lite_prices.parquet'
    ) -> pd.Series:
        """
        加载未来收益
        
        Args:
            prices_path: 价格数据路径
        
        Returns:
            未来收益Series，MultiIndex(date, stock)
        """
        try:
            df = pd.read_parquet(prices_path)
            df['date'] = pd.to_datetime(df['date'])
            
            df = df.sort_values(['stock', 'date'])
            df['future_return'] = df.groupby('stock')['close'].pct_change(1).shift(-1)
            
            result = df.set_index(['date', 'stock'])['future_return']
            return result.dropna()
        except Exception as e:
            raise ValueError(f"Failed to load future return: {e}")
    
    def build_feature_set(
        self,
        formula_df: pd.DataFrame,
        neural_df: pd.DataFrame,
        feature_set_type: str,
        target: pd.Series = None
    ) -> Dict[str, Any]:
        """
        构建指定类型的特征集
        
        Args:
            formula_df: 公式因子DataFrame（必须包含 date, stock 和因子值列）
            neural_df: 神经因子DataFrame
            feature_set_type: 'formula_only', 'neural_only', or 'formula_plus_neural'
            target: 目标变量（用于对齐）
        
        Returns:
            {'features': df_features, 'feature_names': list, 'sample_count': int, 'warnings': []}
        """
        warnings = []
        
        # 获取公式因子列（排除 date, stock）
        formula_factor_cols = [col for col in formula_df.columns if col not in ['date', 'stock']]
        neural_factor_cols = [col for col in neural_df.columns if 'neural_factor_' in col]
        
        if feature_set_type == 'formula_only':
            if len(formula_factor_cols) == 0:
                warnings.append('No formula factor columns found')
            feature_names = formula_factor_cols.copy()
            features = formula_df[['date', 'stock'] + formula_factor_cols].copy()
            
        elif feature_set_type == 'neural_only':
            if len(neural_factor_cols) == 0:
                warnings.append('No neural factor columns found')
            feature_names = neural_factor_cols.copy()
            features = neural_df[['date', 'stock'] + neural_factor_cols].copy()
            
        elif feature_set_type == 'formula_plus_neural':
            if len(formula_factor_cols) == 0:
                warnings.append('No formula factor columns found')
            if len(neural_factor_cols) == 0:
                warnings.append('No neural factor columns found')
            
            formula_features = formula_df[['date', 'stock'] + formula_factor_cols].copy()
            neural_features = neural_df[['date', 'stock'] + neural_factor_cols].copy()
            
            features = pd.merge(
                formula_features,
                neural_features,
                on=['date', 'stock'],
                how='inner'
            )
            feature_names = formula_factor_cols + neural_factor_cols
            
            original_formula_count = len(formula_features)
            original_neural_count = len(neural_features)
            merged_count = len(features)
            
            if merged_count == 0:
                warnings.append('No overlapping samples after merge - formula_plus_neural is empty')
            elif merged_count < min(original_formula_count, original_neural_count) * 0.5:
                warnings.append(f'Large sample drop after merge: formula={original_formula_count}, neural={original_neural_count}, merged={merged_count}')
        
        else:
            raise ValueError(f"Unknown feature_set_type: {feature_set_type}")
        
        # 对齐到目标
        dropped_rows = 0
        dropped_columns = 0
        common_dates = None
        common_stocks = None
        
        if target is not None:
            features['date'] = pd.to_datetime(features['date'])
            features_before = len(features)
            
            features = features.set_index(['date', 'stock'])
            common_idx = features.index.intersection(target.index)
            
            if len(common_idx) == 0:
                warnings.append('No common index with target')
            elif len(common_idx) < len(features) * 0.5:
                warnings.append(f'Significant sample drop during target alignment: {len(features)} -> {len(common_idx)}')
            
            features = features.loc[common_idx]
            dropped_rows = features_before - len(features)
            
            # 统计共同日期和股票
            if len(features) > 0:
                common_dates = sorted(set(features.index.get_level_values('date')))
                common_stocks = sorted(set(features.index.get_level_values('stock')))
        
        return {
            'features': features,
            'feature_names': feature_names,
            'sample_count': len(features),
            'warnings': warnings,
            'feature_set_type': feature_set_type,
            'feature_count': len(feature_names),
            'dropped_rows': dropped_rows,
            'dropped_columns': dropped_columns,
            'common_dates': common_dates,
            'common_stocks': common_stocks
        }
    
    def evaluate_feature_set(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        feature_set_type: str
    ) -> Dict[str, Any]:
        """
        评估特征集的样本外表现
        
        Args:
            features: 特征DataFrame
            target: 目标变量Series
            feature_set_type: 特征集类型
        
        Returns:
            评估结果字典
        """
        if len(features) == 0:
            return {
                'feature_set': feature_set_type,
                'status': 'FAIL',
                'error': 'No samples available'
            }
        
        results = {
            'feature_set': feature_set_type,
            'feature_count': len([col for col in features.columns if col not in ['date', 'stock']]),
            'sample_count': len(features),
            'status': 'OK'
        }
        
        # 计算综合因子（简单平均）
        feature_cols = [col for col in features.columns if col not in ['date', 'stock', 'factor_name']]
        
        if len(feature_cols) > 0:
            features_df = features[feature_cols]
            features_df = (features_df - features_df.mean()) / features_df.std()
            
            composite_factor = features_df.mean(axis=1)
            
            common_idx = composite_factor.index.intersection(target.index)
            
            if len(common_idx) > 10:
                eval_result = self.evaluator.evaluate_single(
                    composite_factor.loc[common_idx],
                    target.loc[common_idx]
                )
                
                results.update({
                    'test_rank_ic': eval_result['rank_ic']['mean'],
                    'test_icir': eval_result.get('icir', 0),
                    'coverage': eval_result.get('coverage', 0),
                    'rank_ic_count': eval_result['rank_ic']['count']
                })
            else:
                results.update({
                    'test_rank_ic': None,
                    'test_icir': None,
                    'coverage': None,
                    'rank_ic_count': 0,
                    'warning': 'Insufficient common samples'
                })
        else:
            results.update({
                'test_rank_ic': None,
                'test_icir': None,
                'coverage': None,
                'rank_ic_count': 0,
                'warning': 'No features available'
            })
        
        return results
    
    def run_comparison(
        self,
        formula_df: pd.DataFrame,
        neural_df: pd.DataFrame,
        target: pd.Series,
        split_info: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        运行三类特征集对比
        
        Args:
            formula_df: 公式因子DataFrame
            neural_df: 神经因子DataFrame
            target: 目标变量Series
            split_info: 切分信息
        
        Returns:
            完整对比结果
        """
        feature_set_types = ['formula_only', 'neural_only', 'formula_plus_neural']
        all_results = []
        
        for feature_set_type in feature_set_types:
            feature_set = self.build_feature_set(
                formula_df, neural_df, feature_set_type, target
            )
            
            if feature_set['sample_count'] > 0:
                eval_result = self.evaluate_feature_set(
                    feature_set['features'], target, feature_set_type
                )
                eval_result['warnings'] = feature_set['warnings']
                eval_result['features'] = feature_set['features']
                eval_result['common_dates'] = feature_set.get('common_dates')
                eval_result['common_stocks'] = feature_set.get('common_stocks')
            else:
                eval_result = {
                    'feature_set': feature_set_type,
                    'feature_count': feature_set['feature_count'],
                    'sample_count': 0,
                    'features': None,
                    'status': 'WARN',
                    'test_rank_ic': None,
                    'test_icir': None,
                    'coverage': None,
                    'warnings': feature_set['warnings'] + ['No samples']
                }
            
            all_results.append(eval_result)
        
        self.comparison_results = {
            'timestamp': datetime.now().isoformat(),
            'split_info': split_info,
            'feature_sets': all_results,
            'can_use_for_live_trading': False
        }
        
        return self.comparison_results
    
    def get_comparison_summary(self) -> pd.DataFrame:
        """
        获取对比结果摘要
        
        Returns:
            对比结果DataFrame
        """
        if not self.comparison_results:
            return pd.DataFrame()
        
        results = []
        for fs in self.comparison_results['feature_sets']:
            results.append({
                'feature_set': fs['feature_set'],
                'feature_count': fs['feature_count'],
                'sample_count': fs['sample_count'],
                'test_rank_ic': fs.get('test_rank_ic'),
                'test_icir': fs.get('test_icir'),
                'coverage': fs.get('coverage'),
                'status': fs.get('status', 'OK'),
                'warnings': ', '.join(fs.get('warnings', []))
            })
        
        return pd.DataFrame(results)
    
    def validate_feature_alignment(
        self,
        formula_df: pd.DataFrame,
        neural_df: pd.DataFrame,
        target: pd.Series
    ) -> Dict[str, Any]:
        """
        验证特征对齐
        
        Args:
            formula_df: 公式因子DataFrame
            neural_df: 神经因子DataFrame
            target: 目标变量Series
        
        Returns:
            验证结果
        """
        issues = []
        
        # 检查日期列
        if 'date' not in formula_df.columns:
            issues.append({'type': 'missing_column', 'column': 'date', 'source': 'formula_df'})
        
        if 'date' not in neural_df.columns:
            issues.append({'type': 'missing_column', 'column': 'date', 'source': 'neural_df'})
        
        if 'stock' not in formula_df.columns:
            issues.append({'type': 'missing_column', 'column': 'stock', 'source': 'formula_df'})
        
        if 'stock' not in neural_df.columns:
            issues.append({'type': 'missing_column', 'column': 'stock', 'source': 'neural_df'})
        
        # 检查目标索引
        if not isinstance(target.index, pd.MultiIndex):
            issues.append({'type': 'invalid_index', 'message': 'target should have MultiIndex(date, stock)'})
        
        # 检查日期范围
        if 'date' in formula_df.columns and 'date' in neural_df.columns:
            formula_dates = set(pd.to_datetime(formula_df['date']))
            neural_dates = set(pd.to_datetime(neural_df['date']))
            
            if not formula_dates & neural_dates:
                issues.append({'type': 'date_mismatch', 'message': 'No overlapping dates'})
        
        # 检查股票范围
        if 'stock' in formula_df.columns and 'stock' in neural_df.columns:
            formula_stocks = set(formula_df['stock'])
            neural_stocks = set(neural_df['stock'])
            
            if not formula_stocks & neural_stocks:
                issues.append({'type': 'stock_mismatch', 'message': 'No overlapping stocks'})
        
        if issues:
            return {
                'status': 'FAIL' if any(issue.get('type') in ['missing_column', 'invalid_index'] for issue in issues) else 'WARN',
                'issues': issues,
                'message': 'Alignment validation issues found'
            }
        else:
            return {
                'status': 'OK',
                'issues': [],
                'message': 'Alignment validation passed'
            }
    
    def check_sample_adequacy(
        self,
        formula_df: pd.DataFrame,
        neural_df: pd.DataFrame,
        test_dates: List[pd.Timestamp],
        test_stocks: List[str]
    ) -> Dict[str, Any]:
        """
        检查样本充分性
        
        Args:
            formula_df: 公式因子DataFrame
            neural_df: 神经因子DataFrame
            test_dates: 测试日期列表
            test_stocks: 测试股票列表
        
        Returns:
            样本充分性检查结果
        """
        formula_dates = set(formula_df['date'])
        neural_dates = set(neural_df['date'])
        common_dates = sorted(formula_dates & neural_dates)
        
        formula_stocks = set(formula_df['stock'])
        neural_stocks = set(neural_df['stock'])
        common_stocks = sorted(formula_stocks & neural_stocks)
        
        test_dates_set = set(test_dates)
        common_test_dates = sorted(test_dates_set & set(common_dates))
        test_trading_days = len(common_test_dates)
        
        test_stocks_set = set(test_stocks)
        common_test_stocks = sorted(test_stocks_set & set(common_stocks))
        test_stock_count = len(common_test_stocks)
        
        dropped_dates = sorted(formula_dates - neural_dates) + sorted(neural_dates - formula_dates)
        dropped_stocks = sorted(formula_stocks - neural_stocks) + sorted(neural_stocks - formula_stocks)
        
        # 计算状态
        status = 'PASS'
        warnings = []
        
        if len(common_dates) < 60:
            status = 'WARN'
            warnings.append(f'Common dates count ({len(common_dates)}) < 60')
        
        if len(common_stocks) < 50:
            status = 'WARN'
            warnings.append(f'Common stocks count ({len(common_stocks)}) < 50')
        
        if test_trading_days < 20:
            status = 'WARN'
            warnings.append(f'Test trading days ({test_trading_days}) < 20')
        
        if test_stock_count < 50:
            status = 'WARN'
            warnings.append(f'Test stock count ({test_stock_count}) < 50')
        
        # 硬性失败条件
        if test_trading_days < 5 or test_stock_count < 10:
            status = 'FAIL'
        
        return {
            'status': status,
            'warnings': warnings,
            'formula_factor_panel_date_range': (formula_df['date'].min(), formula_df['date'].max()),
            'neural_factor_panel_date_range': (neural_df['date'].min(), neural_df['date'].max()),
            'common_date_count': len(common_dates),
            'common_stock_count': len(common_stocks),
            'test_trading_days': test_trading_days,
            'test_stock_count': test_stock_count,
            'dropped_rows': len(formula_df) + len(neural_df),
            'dropped_dates': dropped_dates,
            'dropped_stocks': dropped_stocks,
            'reason_for_dropped_samples': 'Inner join alignment on (date, stock)'
        }
