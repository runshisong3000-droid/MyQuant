"""
Neural Leakage Check - 神经因子未来函数检测

检测神经因子 pipeline 中的潜在未来函数和数据泄露问题。
"""

import pandas as pd
from typing import List, Dict, Any, Tuple
from datetime import datetime


class NeuralLeakageChecker:
    """
    神经因子未来函数检测器

    检查项:
    1. 特征列是否包含未来函数关键词
    2. 序列窗口日期对齐
    3. target日期对齐
    4. scaler fit范围
    """

    def __init__(self):
        self.forbidden_keywords = [
            'future', 'target', 'label', 'return_forward', 'shift_-',
            'lead_', 'next_', 'pred_', 'forecast', 'expected', 'gt_', 'label_'
        ]

    def check_feature_columns(self, columns: List[str]) -> Dict[str, Any]:
        """
        检查特征列是否包含未来函数关键词

        Args:
            columns: 特征列名列表

        Returns:
            {'status': 'OK'/'WARN'/'FAIL', 'issues': [...]}
        """
        issues = []

        for col in columns:
            col_lower = col.lower()
            for keyword in self.forbidden_keywords:
                if keyword in col_lower:
                    issues.append({
                        'type': 'forbidden_keyword',
                        'column': col,
                        'keyword': keyword,
                        'severity': 'high'
                    })

        if issues:
            return {
                'status': 'FAIL',
                'check': 'feature_columns',
                'issues': issues,
                'message': 'Found forbidden keywords in feature columns'
            }

        return {
            'status': 'OK',
            'check': 'feature_columns',
            'issues': [],
            'message': 'All feature columns are safe'
        }

    def check_sequence_dates(
        self,
        df: pd.DataFrame,
        signal_date_col: str = 'signal_date',
        max_date_col: str = 'date'
    ) -> Dict[str, Any]:
        """
        检查序列日期对齐

        Args:
            df: 包含日期信息的DataFrame
            signal_date_col: 信号日期列名
            max_date_col: 输入窗口最大日期列名

        Returns:
            {'status': 'OK'/'WARN'/'FAIL', 'issues': [...]}
        """
        issues = []

        if signal_date_col not in df.columns or max_date_col not in df.columns:
            return {
                'status': 'OK',
                'check': 'sequence_dates',
                'issues': [],
                'message': 'Date columns not found, skipping check'
            }

        violations = df[df[max_date_col] > df[signal_date_col]]

        if len(violations) > 0:
            issues.append({
                'type': 'date_misalignment',
                'count': len(violations),
                'severity': 'high',
                'details': 'Found {} samples where input window extends beyond signal_date'.format(len(violations))
            })

        if issues:
            return {
                'status': 'FAIL',
                'check': 'sequence_dates',
                'issues': issues,
                'message': 'Date misalignment detected'
            }

        return {
            'status': 'OK',
            'check': 'sequence_dates',
            'issues': [],
            'message': 'All sequences properly aligned'
        }

    def check_target_alignment(
        self,
        df: pd.DataFrame,
        signal_date_col: str = 'signal_date',
        target_start_col: str = 'target_start_date'
    ) -> Dict[str, Any]:
        """
        检查 target 日期对齐

        Args:
            df: 包含日期信息的DataFrame
            signal_date_col: 信号日期列名
            target_start_col: target开始日期列名

        Returns:
            {'status': 'OK'/'WARN'/'FAIL', 'issues': [...]}
        """
        issues = []

        if signal_date_col not in df.columns or target_start_col not in df.columns:
            return {
                'status': 'OK',
                'check': 'target_alignment',
                'issues': [],
                'message': 'Target columns not found, skipping check'
            }

        violations = df[df[target_start_col] <= df[signal_date_col]]

        if len(violations) > 0:
            issues.append({
                'type': 'target_before_signal',
                'count': len(violations),
                'severity': 'high',
                'details': 'Found {} samples where target_date is not after signal_date'.format(len(violations))
            })

        if issues:
            return {
                'status': 'FAIL',
                'check': 'target_alignment',
                'issues': issues,
                'message': 'Target alignment violation detected'
            }

        return {
            'status': 'OK',
            'check': 'target_alignment',
            'issues': [],
            'message': 'Target properly aligned after signal_date'
        }

    def check_scaler_fit_scope(
        self,
        train_dates: Tuple[Any, Any],
        val_dates: Tuple[Any, Any],
        test_dates: Tuple[Any, Any]
    ) -> Dict[str, Any]:
        """
        检查 scaler fit 范围

        Args:
            train_dates: 训练集日期范围 (start, end)
            val_dates: 验证集日期范围
            test_dates: 测试集日期范围

        Returns:
            {'status': 'OK'/'WARN'/'FAIL', 'issues': [...]}
        """
        issues = []

        train_start, train_end = train_dates
        val_start, val_end = val_dates
        test_start, test_end = test_dates

        if val_start < train_end:
            issues.append({
                'type': 'val_overlaps_train',
                'severity': 'medium',
                'details': 'Validation period overlaps with training period'
            })

        if test_start < val_end:
            issues.append({
                'type': 'test_overlaps_val',
                'severity': 'medium',
                'details': 'Test period overlaps with validation period'
            })

        if issues:
            return {
                'status': 'WARN',
                'check': 'scaler_fit_scope',
                'issues': issues,
                'message': 'Time periods may have overlap issues'
            }

        return {
            'status': 'OK',
            'check': 'scaler_fit_scope',
            'issues': [],
            'message': 'Time periods properly separated'
        }

    def run_all_checks(
        self,
        columns: List[str] = None,
        metadata: pd.DataFrame = None,
        train_dates: Tuple[Any, Any] = None,
        val_dates: Tuple[Any, Any] = None,
        test_dates: Tuple[Any, Any] = None
    ) -> Dict[str, Any]:
        """
        运行所有检查

        Args:
            columns: 特征列名列表
            metadata: 样本元数据DataFrame
            train_dates: 训练集日期范围
            val_dates: 验证集日期范围
            test_dates: 测试集日期范围

        Returns:
            包含所有检查结果的字典
        """
        results = {}

        if columns is not None:
            results['feature_columns'] = self.check_feature_columns(columns)
        else:
            results['feature_columns'] = {'status': 'SKIP', 'message': 'Not provided'}

        if metadata is not None:
            results['sequence_dates'] = self.check_sequence_dates(metadata)
            results['target_alignment'] = self.check_target_alignment(metadata)
        else:
            results['sequence_dates'] = {'status': 'SKIP', 'message': 'Not provided'}
            results['target_alignment'] = {'status': 'SKIP', 'message': 'Not provided'}

        if train_dates is not None and val_dates is not None and test_dates is not None:
            results['scaler_fit_scope'] = self.check_scaler_fit_scope(
                train_dates, val_dates, test_dates
            )
        else:
            results['scaler_fit_scope'] = {'status': 'SKIP', 'message': 'Not provided'}

        overall_status = 'OK'
        for check_result in results.values():
            if check_result.get('status') == 'FAIL':
                overall_status = 'FAIL'
                break
            elif check_result.get('status') == 'WARN' and overall_status != 'FAIL':
                overall_status = 'WARN'

        results['overall_status'] = overall_status

        return results
