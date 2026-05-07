"""
神经网络隐因子 - Neural Factor

核心功能:
    - 从时间序列中学习隐因子
    - Transformer/LSTM特征提取
    - 因子嵌入
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime


class NeuralFactorExtractor(nn.Module):
    """
    神经网络隐因子提取器
    
    使用深度学习模型从时间序列中学习隐因子：
    - LSTM提取时序特征
    - Transformer提取长期依赖
    - 输出隐向量作为因子
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 32,
        model_type: str = 'lstm'
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.model_type = model_type
        
        if model_type == 'lstm':
            self.backbone = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True
            )
        elif model_type == 'gru':
            self.backbone = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True
            )
        elif model_type == 'transformer':
            self.backbone = nn.Transformer(
                d_model=input_dim,
                nhead=4,
                num_encoder_layers=num_layers
            )
            self.pos_encoder = PositionalEncoding(input_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 if model_type != 'transformer' else input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入序列 [batch, seq_len, input_dim]
            
        Returns:
            隐因子 [batch, output_dim]
        """
        if self.model_type == 'transformer':
            x = self.pos_encoder(x)
            output = self.backbone(x, x)
            output = output.mean(dim=1)
        else:
            _, (h_n, _) = self.backbone(x)
            h_n = torch.cat([h_n[-2, :, :], h_n[-1, :, :]], dim=1)
            output = h_n
        
        return self.fc(output)


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x


class NeuralFactorGenerator:
    """
    神经网络因子生成器
    
    训练神经网络并提取隐因子
    """

    def __init__(self, model_type: str = 'lstm'):
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
    
    def prepare_data(
        self,
        data: pd.DataFrame,
        seq_len: int = 20,
        features: List[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        准备训练数据
        
        Args:
            data: 原始数据
            seq_len: 序列长度
            features: 特征列名
            
        Returns:
            (输入张量, 目标张量)
        """
        features = features or ['open', 'high', 'low', 'close', 'volume', 'amount', 'turnover']
        
        data = data[features].copy()
        
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(data)
        
        X = []
        y = []
        
        for i in range(len(scaled_data) - seq_len):
            X.append(scaled_data[i:i+seq_len])
            y.append(scaled_data[i+seq_len, :])
        
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
    def train(
        self,
        data: pd.DataFrame,
        seq_len: int = 20,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 0.001
    ):
        """
        训练模型
        
        Args:
            data: 训练数据
            seq_len: 序列长度
            epochs: 训练轮数
            batch_size: 批次大小
            lr: 学习率
        """
        X, y = self.prepare_data(data, seq_len)
        
        input_dim = X.shape[-1]
        
        self.model = NeuralFactorExtractor(
            input_dim=input_dim,
            model_type=self.model_type
        ).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.6f}")
    
    def extract_factors(
        self,
        data: pd.DataFrame,
        seq_len: int = 20,
        features: List[str] = None
    ) -> pd.DataFrame:
        """
        提取隐因子
        
        Args:
            data: 数据
            seq_len: 序列长度
            features: 特征列名
            
        Returns:
            隐因子矩阵
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        features = features or ['open', 'high', 'low', 'close', 'volume', 'amount', 'turnover']
        data_values = self.scaler.transform(data[features])
        
        X = []
        for i in range(len(data_values) - seq_len + 1):
            X.append(data_values[i:i+seq_len])
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            factors = self.model(X_tensor).cpu().numpy()
        
        index = data.index[seq_len-1:]
        factor_names = [f'neural_factor_{i+1}' for i in range(factors.shape[1])]
        
        return pd.DataFrame(factors, index=index, columns=factor_names)
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'model_type': self.model_type
        }, path)
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model_type = checkpoint['model_type']
        self.scaler = checkpoint['scaler']
        
        self.model = NeuralFactorExtractor(model_type=self.model_type).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()