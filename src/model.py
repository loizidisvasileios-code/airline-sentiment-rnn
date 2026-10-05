"""Stacked bidirectional LSTM with an attention mechanism, for sentence classification."""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.config import PAD_IDX


class BiLSTMAttention(nn.Module):
    """Embedding, stacked Bi-LSTM, masked attention pooling and a linear classifier."""

    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, output_dim=3,
                 num_layers=2, dropout=0.2):
        super().__init__()

        # padding_idx keeps the <pad> vector fixed at zero and excluded from the gradient
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,  # Applied between the stacked layers, not inside a layer
            bidirectional=True,
        )

        lstm_output_dim = hidden_dim * 2  # Forward and backward states are concatenated

        # Scores one scalar per time step, which softmax turns into attention weights
        self.attn = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, output_dim)

    def forward(self, x):
        """x holds padded word indices of shape (batch, seq_len)."""
        mask = x != PAD_IDX  # True where a real word sits
        lengths = mask.sum(dim=1)  # Real length of every sequence in the batch

        embedded = self.embedding(x)

        # Packing hides the padding from the LSTM. It matters most for the backward
        # direction, which would otherwise start by reading the trailing padding.
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed)
        # total_length pins the output back to the full padded width, so it still lines up
        # with the mask even when the longest sequence in the batch is shorter
        lstm_out, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=x.size(1)
        )

        energy = self.attn(lstm_out).squeeze(-1)  # (batch, seq_len)
        # Padded positions are pushed to a large negative value, so softmax gives them ~0
        energy = energy.masked_fill(~mask, -1e9)
        weights = torch.softmax(energy, dim=1).unsqueeze(-1)

        context = (lstm_out * weights).sum(dim=1)  # Weighted sum over time
        return self.fc(self.dropout(context))
