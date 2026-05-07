"""
Auto Factor Learning Pipeline

核心功能:
    - 自动生成候选因子
    - 自动评估因子
    - 自动审核因子
    - 自动筛选因子
    - 自动保存有效因子
"""

from .factor_candidate import FactorCandidate, FactorStore, FactorRegistry
from .expression_generator import ExpressionGenerator
from .enhanced_generator import EnhancedFactorGenerator
from .genetic_generator import GeneticGenerator
from .neural_factor import NeuralFactorExtractor, NeuralFactorGenerator
from .factor_evaluator import FactorEvaluator
from .factor_gatekeeper import FactorGatekeeper, FactorApprovalPolicy
from .factor_screener import (
    FactorScreener,
    LightGBMImportanceScreener,
    CorrelationFilter,
    StabilityFilter,
    ComprehensiveScreener
)
from .auto_factor_pipeline import AutoFactorPipeline, AutoFactorPipelineBuilder

__all__ = [
    "FactorCandidate",
    "FactorStore",
    "FactorRegistry",
    "ExpressionGenerator",
    "EnhancedFactorGenerator",
    "GeneticGenerator",
    "NeuralFactorExtractor",
    "NeuralFactorGenerator",
    "FactorEvaluator",
    "FactorGatekeeper",
    "FactorApprovalPolicy",
    "FactorScreener",
    "LightGBMImportanceScreener",
    "CorrelationFilter",
    "StabilityFilter",
    "ComprehensiveScreener",
    "AutoFactorPipeline",
    "AutoFactorPipelineBuilder"
]