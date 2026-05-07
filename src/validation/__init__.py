"""
验证模块 - Validation

核心功能:
    - Purged K-Fold交叉验证
    - Walk Forward验证
    - 数据泄露检测
    - 幸存者偏差检测
    - 因子衰减检测
    - 过拟合检测
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Dict, List, Optional, Tuple, Any, Generator
from datetime import datetime, timedelta


class PurgedKFold(BaseCrossValidator):
    """
    Purged K-Fold交叉验证
    
    解决时间序列交叉验证中的数据泄露问题：
    - 训练集和测试集之间需要有一个"清洗期"（purge period）
    - 避免未来信息泄露
    
    参考: Marcos Lopez de Prado的论文
    """

    def __init__(self, n_splits: int = 5, purge_period: int = 5):
        """
        Args:
            n_splits: 折数
            purge_period: 清洗期长度（天数）
        """
        self.n_splits = n_splits
        self.purge_period = purge_period
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
    
    def split(self, X, y=None, groups=None):
        """
        生成训练/测试索引
        
        Args:
            X: 特征数据
            y: 目标数据
            groups: 分组数据
            
        Yields:
            (train_indices, test_indices)
        """
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            dates = X.index.get_level_values(0) if isinstance(X.index, pd.MultiIndex) else X.index
            unique_dates = sorted(dates.unique())
        else:
            unique_dates = np.arange(len(X))
        
        n_dates = len(unique_dates)
        fold_size = n_dates // self.n_splits
        
        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_dates
            
            train_end = max(0, test_start - self.purge_period)
            train_start = 0
            
            if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
                train_mask = dates.isin(unique_dates[train_start:train_end])
                test_mask = dates.isin(unique_dates[test_start:test_end])
            else:
                train_mask = np.arange(len(X)) < train_end
                test_mask = (np.arange(len(X)) >= test_start) & (np.arange(len(X)) < test_end)
            
            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]
            
            yield train_indices, test_indices


class WalkForwardCV(BaseCrossValidator):
    """
    Walk Forward验证
    
    滚动训练和验证：
    - 使用过去的窗口训练模型
    - 在未来的窗口验证
    - 逐步向前滚动
    
    更贴近真实交易场景
    """

    def __init__(self, train_window: int = 252, test_window: int = 63):
        """
        Args:
            train_window: 训练窗口长度（天数）
            test_window: 测试窗口长度（天数）
        """
        self.train_window = train_window
        self.test_window = test_window
    
    def get_n_splits(self, X=None, y=None, groups=None):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            dates = X.index.get_level_values(0) if isinstance(X.index, pd.MultiIndex) else X.index
            n_dates = len(dates.unique())
            return max(0, (n_dates - self.train_window) // self.test_window)
        return 0
    
    def split(self, X, y=None, groups=None):
        """
        生成训练/测试索引
        
        Yields:
            (train_indices, test_indices)
        """
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            dates = X.index.get_level_values(0) if isinstance(X.index, pd.MultiIndex) else X.index
            unique_dates = sorted(dates.unique())
        else:
            unique_dates = np.arange(len(X))
        
        n_dates = len(unique_dates)
        start_idx = self.train_window
        
        while start_idx + self.test_window <= n_dates:
            train_end = start_idx
            train_start = max(0, train_end - self.train_window)
            test_end = start_idx + self.test_window
            
            if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
                train_mask = dates.isin(unique_dates[train_start:train_end])
                test_mask = dates.isin(unique_dates[start_idx:test_end])
            else:
                train_mask = np.arange(len(X)) < train_end
                test_mask = (np.arange(len(X)) >= start_idx) & (np.arange(len(X)) < test_end)
            
            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]
            
            yield train_indices, test_indices
            
            start_idx += self.test_window


class LeakageDetector:
    """
    数据泄露检测器
    
    检测以下类型的泄露：
    1. 未来数据泄露
    2. 标签泄露
    3. 幸存者偏差
    4. 数据窥探
    """

    def __init__(self):
        pass
    
    def detect_future_leakage(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        time_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        检测未来数据泄露
        
        Args:
            X: 特征数据
            y: 目标数据
            time_col: 时间列名
            
        Returns:
            检测结果
        """
        results = {
            'has_leakage': False,
            'warnings': []
        }
        
        if isinstance(X.index, pd.MultiIndex):
            feature_dates = X.index.get_level_values(0)
        else:
            feature_dates = X.index
        
        if isinstance(y.index, pd.MultiIndex):
            target_dates = y.index.get_level_values(0)
        else:
            target_dates = y.index
        
        if time_col is not None and time_col in X.columns:
            feature_times = X[time_col]
            if feature_times.max() > target_dates.max():
                results['has_leakage'] = True
                results['warnings'].append("Feature time column contains future dates")
        
        if feature_dates.max() > target_dates.max():
            results['has_leakage'] = True
            results['warnings'].append("Feature dates extend beyond target dates")
        
        return results
    
    def detect_label_leakage(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        检测标签泄露
        
        Args:
            X: 特征数据
            y: 目标数据
            
        Returns:
            检测结果
        """
        results = {
            'has_leakage': False,
            'high_corr_features': [],
            'max_correlation': 0.0
        }
        
        correlations = X.corrwith(y)
        high_corr = correlations[abs(correlations) > 0.9]
        
        if len(high_corr) > 0:
            results['has_leakage'] = True
            results['high_corr_features'] = high_corr.index.tolist()
            results['max_correlation'] = correlations.max()
        
        return results
    
    def detect_survivorship_bias(
        self,
        X: pd.DataFrame,
        stock_universe: List[str],
        date_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """
        检测幸存者偏差
        
        Args:
            X: 特征数据
            stock_universe: 完整股票池
            date_range: 分析时间范围
            
        Returns:
            检测结果
        """
        results = {
            'has_bias': False,
            'missing_stocks': [],
            'coverage_ratio': 0.0
        }
        
        if isinstance(X.index, pd.MultiIndex):
            stocks_in_data = set(X.index.get_level_values(1).unique())
        else:
            stocks_in_data = set(X.columns)
        
        missing_stocks = [s for s in stock_universe if s not in stocks_in_data]
        
        results['missing_stocks'] = missing_stocks
        results['coverage_ratio'] = len(stocks_in_data) / len(stock_universe)
        
        if len(missing_stocks) > len(stock_universe) * 0.1:
            results['has_bias'] = True
        
        return results
    
    def run_full_detection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        stock_universe: List[str],
        date_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """
        运行完整的泄露检测
        
        Args:
            X: 特征数据
            y: 目标数据
            stock_universe: 完整股票池
            date_range: 分析时间范围
            
        Returns:
            检测结果
        """
        results = {
            'future_leakage': self.detect_future_leakage(X, y),
            'label_leakage': self.detect_label_leakage(X, y),
            'survivorship_bias': self.detect_survivorship_bias(X, y, stock_universe, date_range),
            'overall_status': 'PASS'
        }
        
        if any([
            results['future_leakage']['has_leakage'],
            results['label_leakage']['has_leakage'],
            results['survivorship_bias']['has_bias']
        ]):
            results['overall_status'] = 'FAIL'
        
        return results


class OverfitDetector:
    """
    过拟合检测器
    
    检测模型是否过拟合：
    1. 训练集和测试集表现差异
    2. 样本内/样本外对比
    3. 因子IC稳定性
    """

    def __init__(self):
        pass
    
    def detect_overfit(
        self,
        train_metrics: Dict[str, float],
        test_metrics: Dict[str, float],
        threshold: float = 0.3
    ) -> Dict[str, Any]:
        """
        检测过拟合
        
        Args:
            train_metrics: 训练集指标
            test_metrics: 测试集指标
            threshold: 差异阈值
            
        Returns:
            检测结果
        """
        results = {
            'is_overfitted': False,
            'differences': {},
            'max_difference': 0.0
        }
        
        max_diff = 0.0
        
        for metric in train_metrics:
            if metric in test_metrics:
                train_val = train_metrics[metric]
                test_val = test_metrics[metric]
                
                if train_val != 0:
                    diff = abs((train_val - test_val) / train_val)
                else:
                    diff = abs(train_val - test_val)
                
                results['differences'][metric] = diff
                max_diff = max(max_diff, diff)
        
        results['max_difference'] = max_diff
        
        if max_diff > threshold:
            results['is_overfitted'] = True
        
        return results
    
    def analyze_ic_stability(self, ic_series: pd.Series, window_size: int = 60) -> Dict[str, Any]:
        """
        分析IC稳定性
        
        Args:
            ic_series: IC序列
            window_size: 滚动窗口大小
            
        Returns:
            稳定性分析结果
        """
        rolling_mean = ic_series.rolling(window_size).mean()
        rolling_std = ic_series.rolling(window_size).std()
        
        results = {
            'mean_ic': ic_series.mean(),
            'std_ic': ic_series.std(),
            'rolling_mean_std': rolling_mean.std(),
            'rolling_std_mean': rolling_std.mean(),
            'is_stable': rolling_std.mean() < abs(ic_series.mean()) * 0.5
        }
        
        return results


class FactorDecayChecker:
    """
    因子衰减检查器
    
    检查因子的时效性：
    - 因子IC随时间的衰减
    - 因子有效性的稳定性
    """

    def __init__(self):
        pass
    
    def calculate_decay(self, factor_data: pd.Series, returns: pd.Series, max_lag: int = 20) -> pd.DataFrame:
        """
        计算因子衰减
        
        Args:
            factor_data: 因子数据
            returns: 收益率数据
            max_lag: 最大滞后期数
            
        Returns:
            衰减曲线
        """
        decay = []
        
        for lag in range(max_lag + 1):
            from src.research.factor_test import FactorTest
            factor_test = FactorTest(factor_data, returns)
            ic_summary = factor_test.calculate_ic_summary(lookahead_periods=lag)
            
            decay.append({
                'lag': lag,
                'mean_ic': ic_summary['mean_ic'],
                'ir': ic_summary['ir']
            })
        
        return pd.DataFrame(decay).set_index('lag')
    
    def check_decay_significance(self, decay_df: pd.DataFrame) -> Dict[str, Any]:
        """
        检查衰减的显著性
        
        Args:
            decay_df: 衰减曲线数据
            
        Returns:
            检查结果
        """
        first_ic = decay_df.loc[1, 'mean_ic'] if 1 in decay_df.index else 0
        last_ic = decay_df.iloc[-1]['mean_ic']
        
        half_life_lag = None
        target_ic = abs(first_ic) * 0.5
        
        for lag in decay_df.index:
            if abs(decay_df.loc[lag, 'mean_ic']) <= target_ic:
                half_life_lag = lag
                break
        
        results = {
            'initial_ic': first_ic,
            'final_ic': last_ic,
            'decay_ratio': abs(last_ic) / abs(first_ic) if first_ic != 0 else 0,
            'half_life_lag': half_life_lag,
            'is_persistent': abs(last_ic) >= abs(first_ic) * 0.5
        }
        
        return results


class ValidationReport:
    """
    验证报告生成器
    
    生成完整的验证报告
    """

    def __init__(self):
        pass
    
    def generate_report(
        self,
        leakage_results: Dict[str, Any],
        overfit_results: Dict[str, Any],
        decay_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        生成验证报告
        
        Args:
            leakage_results: 泄露检测结果
            overfit_results: 过拟合检测结果
            decay_results: 衰减检测结果
            
        Returns:
            验证报告
        """
        report = {
            'generated_at': datetime.now().isoformat(),
            'overall_status': 'PASS',
            'sections': []
        }
        
        report['sections'].append({
            'name': 'Leakage Detection',
            'status': 'PASS' if not any([
                leakage_results['future_leakage']['has_leakage'],
                leakage_results['label_leakage']['has_leakage'],
                leakage_results['survivorship_bias']['has_bias']
            ]) else 'FAIL',
            'details': leakage_results
        })
        
        report['sections'].append({
            'name': 'Overfitting Detection',
            'status': 'PASS' if not overfit_results['is_overfitted'] else 'FAIL',
            'details': overfit_results
        })
        
        report['sections'].append({
            'name': 'Factor Decay Analysis',
            'status': 'PASS' if decay_results['is_persistent'] else 'WARNING',
            'details': decay_results
        })
        
        if any(s['status'] == 'FAIL' for s in report['sections']):
            report['overall_status'] = 'FAIL'
        elif any(s['status'] == 'WARNING' for s in report['sections']):
            report['overall_status'] = 'WARNING'
        
        return report