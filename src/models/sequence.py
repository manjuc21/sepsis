"""LSTM sequence model: per-timestep sepsis risk from a padded/masked hourly
patient sequence. Handles variable-length stays via masking, not truncation."""

from __future__ import annotations

import torch
from torch import nn

from src import config


class SepsisLSTM(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int = config.LSTM_HIDDEN_SIZE,
        num_layers: int = config.LSTM_NUM_LAYERS,
        dropout: float = config.LSTM_DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, n_features), lengths: (batch,) true lengths.

        Returns per-timestep logits (batch, seq_len) — padded positions are
        computed but must be masked out by the caller's loss function.
        """
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=x.size(1)
        )
        out = self.dropout(out)
        logits = self.classifier(out).squeeze(-1)
        return logits


def masked_bce_loss(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, pos_weight: torch.Tensor
) -> torch.Tensor:
    """Binary cross-entropy over real (non-padded) timesteps only."""
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    per_step_loss = loss_fn(logits, targets)
    masked = per_step_loss * mask.float()
    return masked.sum() / mask.float().sum().clamp(min=1.0)
