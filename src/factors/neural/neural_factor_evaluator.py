"""
Neural Factor Evaluator - 神经因子评价器

复用现有 FactorEvaluator 评价神经因子。
"""

import pandas as pd
from typing import Dict, Any, List, Optional
import numpy as np


class NeuralFactorEvaluator:
    """
    神经因子评价器

    复用现有 FactorEvaluator 对 neural factors 进行评价。
    """

    def __init__(self):
        from src.factors.auto.factor_evaluator import FactorEvaluator
        self.evaluator = FactorEvaluator()

    def evaluate_neural_factors(
        self,
        factors_input,
        returns: pd.Series,
        factor_names: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        评价所有神经因子

        Args:
            factors_input: 可以是 DataFrame 或 Dict[str, pd.Series]
            returns: 未来收益率序列
            factor_names: 要评价的因子列表，默认评价所有neural_factor_*

        Returns:
            {factor_name: evaluation_result}
        """
        if isinstance(factors_input, dict):
            factors_dict = factors_input
        else:
            factors_df = factors_input
            if factor_names is None:
                factor_names = [col for col in factors_df.columns if col.startswith('neural_factor_')]

            factors_dict = {}
            for factor_name in factor_names:
                if factor_name in factors_df.columns:
                    factors_dict[factor_name] = factors_df.set_index(['date', 'stock'])[factor_name]

        results = {}

        for factor_name, factor_data in factors_dict.items():
            try:
                eval_result = self.evaluator.evaluate_single(factor_data, returns)

                results[factor_name] = {
                    'rank_ic_mean': eval_result['rank_ic']['mean'],
                    'icir': eval_result.get('icir', 0),
                    'coverage': eval_result.get('coverage', 0),
                    'turnover': eval_result.get('turnover', 0),
                    'status': 'evaluated'
                }

            except Exception as e:
                results[factor_name] = {
                    'status': 'error',
                    'error': str(e)
                }

        return results

    def create_evaluation_summary(
        self,
        evaluation_results: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        创建评价汇总表

        Args:
            evaluation_results: evaluate_neural_factors 的结果

        Returns:
            汇总DataFrame
        """
        records = []

        for factor_name, result in evaluation_results.items():
            if result.get('status') == 'evaluated':
                records.append({
                    'factor': factor_name,
                    'rank_ic_mean': result.get('rank_ic_mean', 0),
                    'icir': result.get('icir', 0),
                    'coverage': result.get('coverage', 0),
                    'turnover': result.get('turnover', 0),
                    'status': result.get('status', 'unknown')
                })

        return pd.DataFrame(records).sort_values('rank_ic_mean', ascending=False)

    def get_passing_factors(
        self,
        evaluation_results: Dict[str, Dict[str, Any]],
        min_icir: float = 0.1,
        min_rank_ic: float = 0.01,
        min_coverage: float = 0.5
    ) -> List[str]:
        """
        获取通过筛选的因子

        Args:
            evaluation_results: 评价结果
            min_icir: 最小ICIR
            min_rank_ic: 最小RankIC
            min_coverage: 最小覆盖率

        Returns:
            通过筛选的因子名列表
        """
        passing = []

        for factor_name, result in evaluation_results.items():
            if result.get('status') != 'evaluated':
                continue

            if abs(result.get('rank_ic_mean', 0)) < min_rank_ic:
                continue
            if abs(result.get('icir', 0)) < min_icir:
                continue
            if result.get('coverage', 0) < min_coverage:
                continue

            passing.append(factor_name)

        return passing
