"""
自动因子流水线 - Auto Factor Pipeline

核心功能:
    - 生成候选因子
    - 计算因子值
    - 评估因子
    - 通过审核
    - 保存到因子存储
    - 更新因子注册表
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import os


class AutoFactorPipeline:
    """
    自动因子学习流水线
    
    完整流程：
    1. 生成候选因子 → 2. 计算因子值 → 3. 评估 → 4. 通过gatekeeper → 5. 保存到factor_store → 6. 更新factor_registry
    """

    def __init__(
        self,
        expression_generator,
        genetic_generator,
        neural_factor_generator,
        factor_evaluator,
        factor_gatekeeper,
        factor_store,
        factor_registry
    ):
        self.expression_generator = expression_generator
        self.genetic_generator = genetic_generator
        self.neural_factor_generator = neural_factor_generator
        self.factor_evaluator = factor_evaluator
        self.factor_gatekeeper = factor_gatekeeper
        self.factor_store = factor_store
        self.factor_registry = factor_registry
        
        self.candidates = []
        self.approved_factors = []
        self.rejected_factors = []
    
    def generate_candidates(
        self,
        n_expressions: int = 50,
        use_genetic: bool = False,
        use_neural: bool = False,
        data: Optional[pd.DataFrame] = None,
        returns: Optional[pd.Series] = None
    ):
        """
        生成候选因子
        
        Args:
            n_expressions: 生成数量
            use_genetic: 是否使用遗传编程
            use_neural: 是否使用神经网络
            data: 数据（用于遗传编程和神经网络）
            returns: 收益率数据（用于遗传编程）
        """
        from .factor_candidate import FactorCandidate
        
        expressions = []
        
        expressions.extend(self.expression_generator.generate_batch(n_expressions))
        
        if use_genetic and data is not None and returns is not None:
            genetic_exprs, _ = self.genetic_generator.run(
                self.factor_evaluator,
                data,
                returns,
                generations=5,
                population_size=30
            )
            expressions.extend(genetic_exprs)
        
        if use_neural and data is not None:
            try:
                self.neural_factor_generator.train(data, epochs=30)
                neural_factors = self.neural_factor_generator.extract_factors(data)
                
                for col in neural_factors.columns:
                    expressions.append(f"neural_factor({col})")
            except Exception as e:
                print(f"Neural factor generation failed: {e}")
        
        expressions = list(set(expressions))
        
        for expr in expressions:
            candidate = FactorCandidate(expression=expr, source="auto")
            self.candidates.append(candidate)
        
        print(f"Generated {len(self.candidates)} candidate factors")
    
    def calculate_factor_values(self, data: pd.DataFrame):
        """
        计算因子值
        
        Args:
            data: 原始数据
        """
        for candidate in self.candidates:
            try:
                factor_data = self.expression_generator.evaluate_expression(
                    candidate.expression,
                    data
                )
                candidate.set_data(factor_data)
            except Exception as e:
                print(f"Failed to calculate {candidate.expression}: {e}")
                candidate.data = pd.Series()
    
    def evaluate_candidates(self, returns: pd.Series):
        """
        评估候选因子
        
        Args:
            returns: 收益率数据
        """
        existing_factors = self._get_existing_factors()
        
        for candidate in self.candidates:
            if candidate.data is not None and len(candidate.data.dropna()) > 0:
                try:
                    evaluation = self.factor_evaluator.evaluate_single(
                        candidate.data,
                        returns
                    )
                    
                    if existing_factors:
                        correlations = self.factor_evaluator.calculate_correlation_with_existing(
                            candidate.data,
                            existing_factors
                        )
                        evaluation['correlations'] = correlations
                        evaluation['max_correlation'] = max(abs(c) for c in correlations.values()) if correlations else 0.0
                    
                    evaluation['group_analysis'] = self.factor_evaluator.calculate_group_returns(
                        candidate.data,
                        returns
                    )
                    evaluation['in_out_sample'] = self.factor_evaluator.evaluate_in_sample_out_of_sample(
                        candidate.data,
                        returns
                    )
                    
                    candidate.set_evaluation_results(evaluation)
                except Exception as e:
                    print(f"Failed to evaluate {candidate.expression}: {e}")
    
    def gatekeep_candidates(self):
        """审核候选因子"""
        existing_factors = self._get_existing_factors()
        
        for candidate in self.candidates:
            if candidate.evaluation_results is not None:
                result = self.factor_gatekeeper.approve_or_reject(
                    candidate,
                    existing_factors
                )
                
                if result['approved']:
                    self.approved_factors.append(candidate)
                else:
                    self.rejected_factors.append(candidate)
        
        print(f"Approved: {len(self.approved_factors)}, Rejected: {len(self.rejected_factors)}")
    
    def save_approved_factors(self):
        """保存已批准的因子"""
        for candidate in self.approved_factors:
            self.factor_store.save_factor(candidate)
            self.factor_registry.register_factor(candidate)
            
            print(f"Saved factor: {candidate.name}")
    
    def run(
        self,
        data: pd.DataFrame,
        returns: pd.Series,
        n_expressions: int = 50,
        use_genetic: bool = False,
        use_neural: bool = False
    ) -> Dict[str, Any]:
        """
        运行完整的自动因子流水线
        
        Args:
            data: 原始数据
            returns: 收益率数据
            n_expressions: 生成数量
            use_genetic: 是否使用遗传编程
            use_neural: 是否使用神经网络
            
        Returns:
            运行结果
        """
        print("=" * 60)
        print("Starting Auto Factor Pipeline")
        print("=" * 60)
        
        self.generate_candidates(n_expressions, use_genetic, use_neural, data, returns)
        
        if len(self.candidates) == 0:
            print("No candidates generated")
            return {'approved': [], 'rejected': []}
        
        print("Calculating factor values...")
        self.calculate_factor_values(data)
        
        print("Evaluating candidates...")
        self.evaluate_candidates(returns)
        
        print("Gatekeeping candidates...")
        self.gatekeep_candidates()
        
        print("Saving approved factors...")
        self.save_approved_factors()
        
        print("=" * 60)
        print("Auto Factor Pipeline completed")
        print("=" * 60)
        
        return {
            'total_candidates': len(self.candidates),
            'approved': [c.factor_id for c in self.approved_factors],
            'rejected': [c.factor_id for c in self.rejected_factors],
            'approved_details': [c.to_dict() for c in self.approved_factors]
        }
    
    def _get_existing_factors(self) -> Dict[str, pd.Series]:
        """获取已有因子"""
        factors = {}
        
        for factor_info in self.factor_registry.list_all_factors():
            factor_id = factor_info['factor_id']
            factor_data = self.factor_store.get_factor_data(factor_id)
            
            if factor_data is not None:
                factors[factor_id] = factor_data
        
        return factors
    
    def get_summary(self) -> Dict[str, Any]:
        """获取流水线摘要"""
        return {
            'total_candidates': len(self.candidates),
            'approved_count': len(self.approved_factors),
            'rejected_count': len(self.rejected_factors),
            'approved_factors': [c.to_dict() for c in self.approved_factors]
        }


class AutoFactorPipelineBuilder:
    """
    自动因子流水线构建器
    
    简化流水线的创建过程
    """

    @classmethod
    def build(cls, config: Optional[Dict[str, Any]] = None) -> AutoFactorPipeline:
        """
        构建自动因子流水线
        
        Args:
            config: 配置字典
            
        Returns:
            AutoFactorPipeline实例
        """
        from .expression_generator import ExpressionGenerator
        from .genetic_generator import GeneticGenerator
        from .neural_factor import NeuralFactorGenerator
        from .factor_evaluator import FactorEvaluator
        from .factor_gatekeeper import FactorGatekeeper
        from .factor_candidate import FactorStore, FactorRegistry
        
        config = config or {}
        
        expression_generator = ExpressionGenerator()
        genetic_generator = GeneticGenerator()
        neural_factor_generator = NeuralFactorGenerator(model_type='lstm')
        factor_evaluator = FactorEvaluator()
        factor_gatekeeper = FactorGatekeeper(config.get('gatekeeper', {}))
        factor_store = FactorStore(config.get('store_path', 'data/factor_store'))
        factor_registry = FactorRegistry(config.get('registry_path', 'data/factor_registry.json'))
        
        return AutoFactorPipeline(
            expression_generator=expression_generator,
            genetic_generator=genetic_generator,
            neural_factor_generator=neural_factor_generator,
            factor_evaluator=factor_evaluator,
            factor_gatekeeper=factor_gatekeeper,
            factor_store=factor_store,
            factor_registry=factor_registry
        )