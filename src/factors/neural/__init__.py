"""
Neural Factor Module - 神经因子模块

从原始OHLCV序列中学习latent representation，并转化为neural factors。

主要组件:
- SequenceDataset: 序列数据构造
- SequenceEncoder: 序列编码器(MLP/CNN/Transformer)
- SequenceAutoEncoder: 自编码器
- NeuralFactorExtractor: embedding转换为因子
- NeuralFactorEvaluator: 因子评价(复用现有体系)
"""

from .sequence_dataset import SequenceDataset
from .sequence_encoder import MLPSequenceEncoder, CNN1DEncoder, TinyTransformerEncoder
from .autoencoder import SequenceAutoEncoder
from .neural_factor_extractor import NeuralFactorExtractor
from .neural_factor_evaluator import NeuralFactorEvaluator

__all__ = [
    "SequenceDataset",
    "MLPSequenceEncoder",
    "CNN1DEncoder",
    "TinyTransformerEncoder",
    "SequenceAutoEncoder",
    "NeuralFactorExtractor",
    "NeuralFactorEvaluator"
]
