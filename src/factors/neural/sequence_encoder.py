"""
Sequence Encoder - 序列编码器

提供三种轻量编码器:
- MLPSequenceEncoder: MLP扁平化编码器
- CNN1DEncoder: 1D卷积编码器
- TinyTransformerEncoder: 轻量Transformer编码器
"""

import torch
import torch.nn as nn


class MLPSequenceEncoder(nn.Module):
    """
    MLP序列编码器

    将序列扁平化后通过MLP编码为embedding。
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, embedding_dim: int = 8, lookback_window: int = 20):
        super().__init__()

        self.lookback_window = lookback_window
        self.input_dim = input_dim
        in_features = lookback_window * input_dim

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim)
        )

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding


class CNN1DEncoder(nn.Module):
    """
    1D卷积序列编码器

    在时间维度上使用1D卷积提取局部模式。
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, embedding_dim: int = 8):
        super().__init__()

        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, embedding_dim)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.activation(x)

        x = self.pool(x).squeeze(-1)

        embedding = self.fc(x)
        return embedding


class TinyTransformerEncoder(nn.Module):
    """
    轻量Transformer编码器

    使用少量层和注意力头进行序列编码。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        embedding_dim: int = 8,
        num_heads: int = 2,
        num_layers: int = 1
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        x = self.input_proj(x)

        x = self.transformer(x)

        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(-1)

        embedding = self.fc(x)
        return embedding


def create_encoder(encoder_type: str, input_dim: int, hidden_dim: int = 32, embedding_dim: int = 8, lookback_window: int = 20):
    """
    工厂函数：创建编码器

    Args:
        encoder_type: 'mlp', 'cnn', 'transformer'
        input_dim: 输入特征维度
        hidden_dim: 隐藏层维度
        embedding_dim: embedding维度
        lookback_window: 序列长度

    Returns:
        Encoder model
    """
    if encoder_type.lower() == 'mlp':
        return MLPSequenceEncoder(input_dim, hidden_dim, embedding_dim, lookback_window)
    elif encoder_type.lower() == 'cnn':
        return CNN1DEncoder(input_dim, hidden_dim, embedding_dim)
    elif encoder_type.lower() == 'transformer':
        return TinyTransformerEncoder(input_dim, hidden_dim, embedding_dim)
    else:
        raise ValueError("Unknown encoder type: {}".format(encoder_type))
