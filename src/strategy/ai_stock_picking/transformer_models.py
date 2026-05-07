"""
Transformer时序模型

功能:
    - 基于Transformer的时序预测模型
    - 多头注意力机制
    - 位置编码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerTimeSeries(nn.Module):
    """Transformer时序预测模型"""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        output_dim: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, src):
        src = self.input_proj(src) * np.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output[:, -1, :]
        return self.fc(output)

    def train_model(
        self,
        train_loader,
        val_loader=None,
        epochs=100,
        lr=0.001,
        weight_decay=0.0,
        patience=10
    ):
        criterion = nn.BCEWithLogitsLoss() if self.fc.out_features == 1 else nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
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
                if self.fc.out_features == 1:
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

                        if self.fc.out_features == 1:
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
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}")

        return history

    def predict(self, X):
        self.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self(X_tensor)
            if self.fc.out_features == 1:
                return torch.sigmoid(outputs).cpu().numpy()
            return outputs.cpu().numpy()

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_proj.in_features,
            'd_model': self.d_model,
            'nhead': self.transformer_encoder.layers[0].self_attn.num_heads,
            'num_layers': len(self.transformer_encoder.layers),
            'output_dim': self.fc.out_features
        }, path)

    @classmethod
    def load_model(cls, path):
        checkpoint = torch.load(path)
        model = cls(
            input_dim=checkpoint['input_dim'],
            d_model=checkpoint['d_model'],
            nhead=checkpoint['nhead'],
            num_layers=checkpoint['num_layers'],
            output_dim=checkpoint['output_dim']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


class TSTransformerEncoder(nn.Module):
    """时间序列Transformer编码器"""

    def __init__(
        self,
        input_dim: int,
        max_seq_len: int = 100,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.d_model = d_model

    def forward(self, x):
        x = self.input_proj(x) * np.sqrt(self.d_model)
        x = self.pos_encoder(x)
        output = self.encoder(x)
        return output


class TimeSeriesTransformer(nn.Module):
    """完整的时间序列Transformer模型"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        max_seq_len: int = 100,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.encoder = TSTransformerEncoder(
            input_dim=input_dim,
            max_seq_len=max_seq_len,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        self.fc = nn.Linear(d_model, output_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        encoded = self.encoder(x)
        last_hidden = encoded[:, -1, :]
        return self.fc(last_hidden)

    def predict(self, X):
        self.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self(X_tensor)
            if self.fc.out_features == 1:
                return torch.sigmoid(outputs).cpu().numpy()
            return outputs.cpu().numpy()