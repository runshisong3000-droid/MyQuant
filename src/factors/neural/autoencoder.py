"""
Sequence AutoEncoder - 序列自编码器

编码输入序列为embedding，并解码重构原始序列。
"""

import torch
import torch.nn as nn
from typing import Tuple


class SequenceAutoEncoder(nn.Module):
    """
    序列自编码器

    编码序列为latent embedding，并重构原始序列。

    Attributes:
        encoder: 编码器网络
        decoder: 解码器网络
        lookback_window: 序列长度
        num_features: 特征数量
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        embedding_dim: int = 8,
        lookback_window: int = 20,
        encoder_type: str = 'mlp'
    ):
        super().__init__()

        self.lookback_window = lookback_window
        self.num_features = input_dim

        if encoder_type.lower() == 'mlp':
            from .sequence_encoder import MLPSequenceEncoder
            self.encoder = MLPSequenceEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim,
                lookback_window=lookback_window
            )
        elif encoder_type.lower() == 'cnn':
            from .sequence_encoder import CNN1DEncoder
            self.encoder = CNN1DEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim
            )
        elif encoder_type.lower() == 'transformer':
            from .sequence_encoder import TinyTransformerEncoder
            self.encoder = TinyTransformerEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim
            )
        else:
            raise ValueError("Unknown encoder type: {}".format(encoder_type))

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, lookback_window * input_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入张量，shape (batch_size, lookback_window, num_features)

        Returns:
            reconstruction: 重构序列，shape (batch_size, lookback_window, num_features)
            embedding: 编码embedding，shape (batch_size, embedding_dim)
        """
        batch_size, seq_len, num_features = x.shape

        embedding = self.encoder(x)

        decoded = self.decoder(embedding)

        reconstruction = decoded.view(batch_size, seq_len, num_features)

        return reconstruction, embedding

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        只编码，不解码

        Args:
            x: 输入张量

        Returns:
            embedding: 编码结果
        """
        return self.encoder(x)
