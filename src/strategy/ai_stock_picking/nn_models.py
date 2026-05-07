"""
深度学习模型框架

功能:
    - PyTorch模型基类
    - 时序预测模型
    - Transformer模型
    - 模型训练和验证
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime


class TimeSeriesDataset(Dataset):
    """时序数据集"""

    def __init__(self, features: np.ndarray, labels: np.ndarray, seq_length: int):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.features) - self.seq_length

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_length]
        y = self.labels[idx + self.seq_length]
        return x, y


class BaseNNModel(nn.Module):
    """神经网络模型基类"""

    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        """前向传播"""
        pass

    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        epochs: int = 100,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        patience: int = 10
    ) -> Dict[str, List[float]]:
        """训练模型"""
        criterion = nn.BCEWithLogitsLoss() if self.output_dim == 1 else nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                outputs = self(x)
                loss = criterion(outputs.squeeze(), y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * x.size(0)
                if self.output_dim == 1:
                    predictions = (torch.sigmoid(outputs) > 0.5).float()
                    train_correct += (predictions.squeeze() == y).sum().item()
                    train_total += y.size(0)

            train_loss /= len(train_loader.dataset)
            train_acc = train_correct / train_total if train_total > 0 else 0

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            if val_loader:
                self.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(self.device), y.to(self.device)
                        outputs = self(x)
                        loss = criterion(outputs.squeeze(), y)
                        val_loss += loss.item() * x.size(0)

                        if self.output_dim == 1:
                            predictions = (torch.sigmoid(outputs) > 0.5).float()
                            val_correct += (predictions.squeeze() == y).sum().item()
                            val_total += y.size(0)

                val_loss /= len(val_loader.dataset)
                val_acc = val_correct / val_total if val_total > 0 else 0

                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_model('best_model.pt')
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
                      f"{', Val Loss: ' + str(val_loss)[:6] + ', Val Acc: ' + str(val_acc)[:6] if val_loader else ''}")

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        self.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self(X_tensor)
            if self.output_dim == 1:
                return torch.sigmoid(outputs).cpu().numpy()
            return outputs.cpu().numpy()

    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'output_dim': self.output_dim
        }, path)

    @classmethod
    def load_model(cls, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        model = cls(checkpoint['input_dim'], checkpoint['output_dim'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


class TimeSeriesMLP(BaseNNModel):
    """多层感知器时序模型"""

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 1):
        super().__init__(input_dim, output_dim)

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        if len(x.shape) > 2:
            x = x.reshape(x.size(0), -1)
        return self.layers(x)


class LSTMModel(BaseNNModel):
    """LSTM时序模型"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2
    ):
        super().__init__(input_dim, output_dim)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        return self.fc(last_hidden)


class GRUModel(BaseNNModel):
    """GRU时序模型"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2
    ):
        super().__init__(input_dim, output_dim)

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_hidden = gru_out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        return self.fc(last_hidden)


class TemporalAttention(nn.Module):
    """时间注意力机制"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        scores = torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(x.size(-1))
        weights = self.softmax(scores)
        output = torch.bmm(weights, values)

        return output, weights


class AttentionLSTM(BaseNNModel):
    """带注意力机制的LSTM"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2
    ):
        super().__init__(input_dim, output_dim)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.attention = TemporalAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attended_out, _ = self.attention(lstm_out)
        last_hidden = attended_out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        return self.fc(last_hidden)