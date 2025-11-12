"""Configurable multivariate LSTM with optional attention and MC Dropout for uncertainty.
Outputs a single value (regression) for t+pred_horizon close price.
"""
import torch
import torch.nn as nn
from .attention import Attention


class LSTMForecast(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2,
                 bidirectional=False, use_attention=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.use_attention = use_attention

        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0,
                            bidirectional=bidirectional)
        out_dim = hidden_dim * (2 if bidirectional else 1)
        if use_attention:
            self.attention = Attention(out_dim)
            self.head = nn.Sequential(
                nn.Linear(out_dim, out_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(out_dim // 2, 1)
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(out_dim, out_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(out_dim // 2, 1)
            )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)  # out: (batch, seq_len, hidden*directions)
        if self.use_attention:
            context, attn_weights = self.attention(out)
            out = self.head(context).squeeze(-1)
            return out, attn_weights
        else:
            # take last time-step
            last = out[:, -1, :]
            out = self.head(last).squeeze(-1)
            return out, None


if __name__ == '__main__':
    import torch
    model = LSTMForecast(input_dim=10)
    x = torch.randn(2, 60, 10)
    y, att = model(x)
    print(y.shape, None if att is None else att.shape)