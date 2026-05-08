"""
OutOfSampleValidator - 样本外验证器

功能:
    - 按时间切分数据为 train / validation / test
    - 验证切分的正确性
    - 防止未来数据泄露
    - 输出结构化 split 信息
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime


class OutOfSampleValidator:
    """
    样本外验证器
    
    负责:
    1. 按时间顺序切分数据
    2. 验证切分的时间顺序正确性
    3. 检测潜在的数据泄露
    4. 输出结构化的切分信息
    """
    
    def __init__(self):
        self.split_info = {}
    
    def split_by_time(
        self,
        data: pd.DataFrame,
        date_column: str = 'date',
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        group_by_stock: bool = True,
        stock_column: str = 'stock'
    ) -> Dict[str, pd.DataFrame]:
        """
        按时间顺序切分数据
        
        Args:
            data: 包含日期列的DataFrame
            date_column: 日期列名
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            group_by_stock: 是否按股票分组
            stock_column: 股票列名
        
        Returns:
            {'train': df_train, 'validation': df_val, 'test': df_test}
        
        Raises:
            ValueError: 如果比例不等于1或数据不足
        """
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
        
        if date_column not in data.columns:
            raise ValueError(f"date_column '{date_column}' not found in data")
        
        df = data.copy()
        
        if group_by_stock and stock_column in df.columns:
            unique_dates = sorted(df[date_column].unique())
        else:
            unique_dates = sorted(df[date_column].unique())
        
        n_dates = len(unique_dates)
        
        if n_dates < 3:
            raise ValueError("至少需要3个不同日期才能切分")
        
        train_end_idx = int(n_dates * train_ratio)
        val_end_idx = int(n_dates * (train_ratio + val_ratio))
        
        if train_end_idx < 1:
            raise ValueError("训练集至少需要1个日期")
        if val_end_idx <= train_end_idx:
            raise ValueError("验证集至少需要1个日期")
        if val_end_idx >= n_dates:
            raise ValueError("测试集至少需要1个日期")
        
        train_dates = set(unique_dates[:train_end_idx])
        val_dates = set(unique_dates[train_end_idx:val_end_idx])
        test_dates = set(unique_dates[val_end_idx:])
        
        df_train = df[df[date_column].isin(train_dates)].copy()
        df_val = df[df[date_column].isin(val_dates)].copy()
        df_test = df[df[date_column].isin(test_dates)].copy()
        
        # 记录切分信息
        self.split_info = {
            'train_start': str(unique_dates[0]),
            'train_end': str(unique_dates[train_end_idx - 1]),
            'validation_start': str(unique_dates[train_end_idx]),
            'validation_end': str(unique_dates[val_end_idx - 1]),
            'test_start': str(unique_dates[val_end_idx]),
            'test_end': str(unique_dates[-1]),
            'train_samples': len(df_train),
            'validation_samples': len(df_val),
            'test_samples': len(df_test),
            'stock_count': df[stock_column].nunique() if stock_column in df.columns else 1,
            'trading_days': {
                'train': len(train_dates),
                'validation': len(val_dates),
                'test': len(test_dates)
            },
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio
        }
        
        return {
            'train': df_train,
            'validation': df_val,
            'test': df_test
        }
    
    def validate_split(self) -> Dict[str, Any]:
        """
        验证切分的正确性
        
        Returns:
            {'status': 'OK'/'WARN'/'FAIL', 'issues': [...], 'message': '...'}
        """
        if not self.split_info:
            return {
                'status': 'FAIL',
                'issues': ['No split info available'],
                'message': 'Please call split_by_time first'
            }
        
        issues = []
        
        # 检查时间顺序
        train_end = self.split_info['train_end']
        val_start = self.split_info['validation_start']
        val_end = self.split_info['validation_end']
        test_start = self.split_info['test_start']
        
        if train_end >= val_start:
            issues.append({
                'type': 'time_order_error',
                'message': f"train_end ({train_end}) >= validation_start ({val_start})",
                'severity': 'high'
            })
        
        if val_end >= test_start:
            issues.append({
                'type': 'time_order_error',
                'message': f"validation_end ({val_end}) >= test_start ({test_start})",
                'severity': 'high'
            })
        
        # 检查样本数
        train_samples = self.split_info['train_samples']
        val_samples = self.split_info['validation_samples']
        test_samples = self.split_info['test_samples']
        
        min_samples = 10
        if train_samples < min_samples:
            issues.append({
                'type': 'sample_size_warning',
                'message': f"训练集样本数不足 ({train_samples} < {min_samples})",
                'severity': 'medium'
            })
        
        if val_samples < min_samples:
            issues.append({
                'type': 'sample_size_warning',
                'message': f"验证集样本数不足 ({val_samples} < {min_samples})",
                'severity': 'medium'
            })
        
        if test_samples < min_samples:
            issues.append({
                'type': 'sample_size_warning',
                'message': f"测试集样本数不足 ({test_samples} < {min_samples})",
                'severity': 'medium'
            })
        
        if any(issue['severity'] == 'high' for issue in issues):
            return {
                'status': 'FAIL',
                'issues': issues,
                'message': 'Critical issues found in split validation'
            }
        elif issues:
            return {
                'status': 'WARN',
                'issues': issues,
                'message': 'Warnings found in split validation'
            }
        else:
            return {
                'status': 'OK',
                'issues': [],
                'message': 'Split validation passed'
            }
    
    def get_split_info(self) -> Dict[str, Any]:
        """
        获取切分信息
        
        Returns:
            结构化的切分信息字典
        """
        return self.split_info
    
    def check_future_leakage(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        date_column: str = 'date'
    ) -> Dict[str, Any]:
        """
        检查是否存在未来数据泄露
        
        Args:
            features: 特征数据
            target: 目标数据
            date_column: 日期列名
        
        Returns:
            {'status': 'OK'/'WARN'/'FAIL', 'issues': [...], 'message': '...'}
        """
        issues = []
        
        # 检查特征列名
        forbidden_keywords = [
            'future', 'target', 'label', 'return_forward', 'shift_-',
            'lead_', 'next_', 'pred_', 'forecast', 'expected', 'gt_'
        ]
        
        for col in features.columns:
            col_lower = col.lower()
            for keyword in forbidden_keywords:
                if keyword in col_lower:
                    issues.append({
                        'type': 'forbidden_keyword',
                        'column': col,
                        'keyword': keyword,
                        'severity': 'high'
                    })
        
        # 检查日期对齐
        if date_column in features.columns and date_column in target.index.names:
            feature_dates = set(features[date_column])
            target_dates = set(target.index.get_level_values(date_column))
            
            if feature_dates > target_dates:
                issues.append({
                    'type': 'date_leakage',
                    'message': 'Feature dates extend beyond target dates',
                    'severity': 'high'
                })
        
        if any(issue['severity'] == 'high' for issue in issues):
            return {
                'status': 'FAIL',
                'issues': issues,
                'message': 'Future leakage detected'
            }
        elif issues:
            return {
                'status': 'WARN',
                'issues': issues,
                'message': 'Potential leakage warnings'
            }
        else:
            return {
                'status': 'OK',
                'issues': [],
                'message': 'No future leakage detected'
            }
