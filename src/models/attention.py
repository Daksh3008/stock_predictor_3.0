"""Simple attention layer for sequence outputs.
Takes LSTM outputs (batch, seq_len, hidden) and returns weighted sum + attn weights.
"""
import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, lstm_outputs):
        # lstm_outputs: (batch, seq_len, hidden)
        scores = self.attn(lstm_outputs)  # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1)  # (batch, seq_len, 1)
        context = (weights * lstm_outputs).sum(dim=1)  # (batch, hidden)
        return context, weights.squeeze(-1)