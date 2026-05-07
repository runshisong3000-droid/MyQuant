"""
因子审核器 - Factor Gatekeeper

核心功能:
    - 设置因子入库规则
    - 自动审核候选因子
    - 决定是否入库
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime


class FactorGatekeeper:
    """
    因子审核器
    
    设置严格的因子入库标准：
    1. 样本外 RankIC 均值 > 0.01
    2. ICIR > 0.3
    3. 与已有因子最大相关性 < 0.7
    4. 多空组合收益为正
    5. 因子覆盖率 > 80%
    6. 不能包含未来函数
    7. 公式复杂度不能太高
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        self.rules = [
            self.check_ic,
            self.check_icir,
            self.check_correlation,
            self.check_long_short,
            self.check_coverage,
            self.check_future_leakage,
            self.check_complexity
        ]
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'min_out_of_sample_ic': 0.01,
            'min_icir': 0.3,
            'max_correlation': 0.7,
            'min_long_short_return': 0.0,
            'min_coverage': 0.8,
            'max_complexity': 10,
            'min_monotonicity': 0.5
        }
    
    def check_ic(self, candidate) -> Dict[str, Any]:
        """检查样本外RankIC"""
        result = {
            'passed': False,
            'score': 0.0,
            'reason': ''
        }
        
        if candidate.evaluation_results is None:
            result['reason'] = "No evaluation results"
            return result
        
        oos_ic = candidate.evaluation_results.get('in_out_sample', {}).get('out_of_sample', {}).get('rank_ic', {}).get('mean', 0)
        
        if abs(oos_ic) >= self.config['min_out_of_sample_ic']:
            result['passed'] = True
            result['score'] = abs(oos_ic)
            result['reason'] = f"OOS IC: {oos_ic:.4f}"
        else:
            result['reason'] = f"OOS IC {oos_ic:.4f} < threshold {self.config['min_out_of_sample_ic']}"
        
        return result
    
    def check_icir(self, candidate) -> Dict[str, Any]:
        """检查ICIR"""
        result = {
            'passed': False,
            'score': 0.0,
            'reason': ''
        }
        
        if candidate.evaluation_results is None:
            result['reason'] = "No evaluation results"
            return result
        
        icir = candidate.evaluation_results.get('icir', 0)
        
        if icir >= self.config['min_icir']:
            result['passed'] = True
            result['score'] = icir
            result['reason'] = f"ICIR: {icir:.4f}"
        else:
            result['reason'] = f"ICIR {icir:.4f} < threshold {self.config['min_icir']}"
        
        return result
    
    def check_correlation(self, candidate, existing_factors: Optional[Dict[str, pd.Series]] = None) -> Dict[str, Any]:
        """检查与已有因子的相关性"""
        result = {
            'passed': True,
            'score': 1.0,
            'reason': ''
        }
        
        if existing_factors is None or len(existing_factors) == 0:
            result['reason'] = "No existing factors"
            return result
        
        if candidate.evaluation_results is None:
            result['reason'] = "No evaluation results"
            result['passed'] = False
            return result
        
        max_corr = candidate.evaluation_results.get('max_correlation', 0)
        
        if max_corr < self.config['max_correlation']:
            result['passed'] = True
            result['score'] = 1 - max_corr
            result['reason'] = f"Max correlation: {max_corr:.4f}"
        else:
            result['passed'] = False
            result['reason'] = f"Max correlation {max_corr:.4f} >= threshold {self.config['max_correlation']}"
        
        return result
    
    def check_long_short(self, candidate) -> Dict[str, Any]:
        """检查多空收益"""
        result = {
            'passed': False,
            'score': 0.0,
            'reason': ''
        }
        
        if candidate.evaluation_results is None:
            result['reason'] = "No evaluation results"
            return result
        
        group_analysis = candidate.evaluation_results.get('group_analysis', {})
        long_short_mean = group_analysis.get('long_short_mean', 0)
        
        if long_short_mean > self.config['min_long_short_return']:
            result['passed'] = True
            result['score'] = long_short_mean
            result['reason'] = f"Long-short mean: {long_short_mean:.4f}"
        else:
            result['reason'] = f"Long-short mean {long_short_mean:.4f} <= threshold {self.config['min_long_short_return']}"
        
        return result
    
    def check_coverage(self, candidate) -> Dict[str, Any]:
        """检查因子覆盖率"""
        result = {
            'passed': False,
            'score': 0.0,
            'reason': ''
        }
        
        coverage = candidate.metadata.get('coverage', 0)
        
        if coverage >= self.config['min_coverage']:
            result['passed'] = True
            result['score'] = coverage
            result['reason'] = f"Coverage: {coverage:.2%}"
        else:
            result['reason'] = f"Coverage {coverage:.2%} < threshold {self.config['min_coverage']:.2%}"
        
        return result
    
    def check_future_leakage(self, candidate) -> Dict[str, Any]:
        """检查是否包含未来函数"""
        result = {
            'passed': True,
            'score': 1.0,
            'reason': ''
        }
        
        expression = candidate.expression
        
        future_keywords = ['future', 'shift\\(-', 'lead', 'forward']
        
        for keyword in future_keywords:
            if keyword in expression.lower():
                result['passed'] = False
                result['score'] = 0.0
                result['reason'] = f"Potential future leakage: {keyword}"
                return result
        
        result['reason'] = "No future leakage detected"
        return result
    
    def check_complexity(self, candidate) -> Dict[str, Any]:
        """检查公式复杂度"""
        result = {
            'passed': False,
            'score': 0.0,
            'reason': ''
        }
        
        expression = candidate.expression
        complexity = expression.count('(')
        
        if complexity <= self.config['max_complexity']:
            result['passed'] = True
            result['score'] = 1 - complexity / self.config['max_complexity']
            result['reason'] = f"Complexity: {complexity}"
        else:
            result['reason'] = f"Complexity {complexity} > threshold {self.config['max_complexity']}"
        
        return result
    
    def check_monotonicity(self, candidate) -> Dict[str, Any]:
        """检查分组单调性"""
        result = {
            'passed': False,
            'score': 0.0,
            'reason': ''
        }
        
        if candidate.evaluation_results is None:
            result['reason'] = "No evaluation results"
            return result
        
        group_analysis = candidate.evaluation_results.get('group_analysis', {})
        monotonicity = group_analysis.get('monotonicity', 0)
        
        if monotonicity >= self.config['min_monotonicity']:
            result['passed'] = True
            result['score'] = monotonicity
            result['reason'] = f"Monotonicity: {monotonicity:.4f}"
        else:
            result['reason'] = f"Monotonicity {monotonicity:.4f} < threshold {self.config['min_monotonicity']}"
        
        return result
    
    def approve_or_reject(
        self,
        candidate,
        existing_factors: Optional[Dict[str, pd.Series]] = None
    ) -> Dict[str, Any]:
        """
        审核因子
        
        Args:
            candidate: 候选因子
            existing_factors: 已有因子
            
        Returns:
            审核结果
        """
        results = {}
        
        for rule in self.rules:
            if rule.__name__ == 'check_correlation':
                results[rule.__name__] = rule(candidate, existing_factors)
            else:
                results[rule.__name__] = rule(candidate)
        
        passed_rules = sum(1 for r in results.values() if r['passed'])
        total_rules = len(results)
        
        overall_pass = all(r['passed'] for r in results.values())
        
        gatekeeper_results = {
            'approved': overall_pass,
            'passed_rules': passed_rules,
            'total_rules': total_rules,
            'rule_results': results,
            'score': self._calculate_overall_score(results)
        }
        
        candidate.set_gatekeeper_results(gatekeeper_results)
        
        return gatekeeper_results
    
    def _calculate_overall_score(self, rule_results: Dict[str, Dict[str, Any]]) -> float:
        """计算综合分数"""
        scores = [r['score'] for r in rule_results.values()]
        
        if len(scores) == 0:
            return 0.0
        
        return np.mean(scores)
    
    def evaluate_multiple(
        self,
        candidates,
        existing_factors: Optional[Dict[str, pd.Series]] = None
    ) -> List[Dict[str, Any]]:
        """
        批量审核候选因子
        
        Args:
            candidates: 候选因子列表
            existing_factors: 已有因子
            
        Returns:
            审核结果列表
        """
        results = []
        
        for candidate in candidates:
            result = self.approve_or_reject(candidate, existing_factors)
            results.append(result)
        
        return results


class FactorApprovalPolicy:
    """
    因子批准策略
    
    定义不同的批准策略：
    - 严格模式：所有规则必须通过
    - 宽松模式：大部分规则通过即可
    - 加权模式：根据规则重要性加权评分
    """

    def __init__(self, mode: str = 'strict'):
        self.mode = mode
        self.weights = {
            'check_ic': 0.25,
            'check_icir': 0.25,
            'check_correlation': 0.20,
            'check_long_short': 0.15,
            'check_coverage': 0.10,
            'check_future_leakage': 0.05,
            'check_complexity': 0.00
        }
    
    def approve(self, gatekeeper_results: Dict[str, Any]) -> bool:
        """根据策略决定是否批准"""
        if self.mode == 'strict':
            return gatekeeper_results['approved']
        
        elif self.mode == 'lenient':
            return gatekeeper_results['passed_rules'] >= gatekeeper_results['total_rules'] * 0.7
        
        elif self.mode == 'weighted':
            score = 0.0
            
            for rule_name, result in gatekeeper_results['rule_results'].items():
                if rule_name in self.weights:
                    score += result['score'] * self.weights[rule_name]
            
            return score >= 0.7
        
        return False