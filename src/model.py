"""The LSTM: column ids -> embedding -> 2 stacked LSTM layers -> scores for the next column."""
import torch
from torch import nn


class LevelLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # `dropout` is applied between the stacked LSTM layers...
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0.0, batch_first=True)
        # ...and again on the last layer's output before the linear layer.
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        """x: (batch, window) column ids -> logits (batch, vocab_size) for the NEXT column."""
        out, _ = self.lstm(self.embed(x))   # out: (batch, window, hidden)
        last = out[:, -1, :]                # the state after reading the whole window
        return self.head(self.drop(last))


def build_model(cfg, vocab_size):
    m = cfg["model"]
    return LevelLSTM(vocab_size, m["embed_dim"], m["hidden_size"], m["num_layers"], m["dropout"])
