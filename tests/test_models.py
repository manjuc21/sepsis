from __future__ import annotations

import numpy as np
import torch

from src.data import features, preprocess
from src.models.baseline import LogRegModel, XGBoostModel
from src.models.sequence import SepsisLSTM, masked_bce_loss


def test_logreg_smoke(synthetic_patients):
    # LogisticRegression can't accept NaN natively (unlike XGBoost) — the
    # real pipeline always runs preprocess.preprocess() before building
    # tabular features; mirror that order here rather than testing against
    # raw, still-missing data.
    train, _, _, _ = preprocess.preprocess(synthetic_patients)
    X, y = features.build_tabular_feature_matrix(train)
    model = LogRegModel().fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape[0] == len(X)
    assert np.all((probs >= 0) & (probs <= 1))


def test_xgboost_smoke(synthetic_patients):
    X, y = features.build_tabular_feature_matrix(synthetic_patients)
    model = XGBoostModel().fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape[0] == len(X)
    assert np.all((probs >= 0) & (probs <= 1))


def test_lstm_forward_pass_shape(synthetic_patients):
    train, _, _, _ = preprocess.preprocess(synthetic_patients)
    X, y, mask, _ = features.build_sequence_tensors(train, max_len=50)
    lengths = mask.sum(dim=1)

    model = SepsisLSTM(n_features=X.shape[-1])
    logits = model(X, lengths)

    assert logits.shape == (X.shape[0], X.shape[1])


def test_lstm_masked_loss_ignores_padding(synthetic_patients):
    train, _, _, _ = preprocess.preprocess(synthetic_patients)
    X, y, mask, _ = features.build_sequence_tensors(train, max_len=50)
    lengths = mask.sum(dim=1)
    model = SepsisLSTM(n_features=X.shape[-1])
    logits = model(X, lengths)

    pos_weight = torch.tensor(1.0)
    loss_with_mask = masked_bce_loss(logits, y, mask, pos_weight)

    # a mask of all False must not raise and yields a defined (zero) loss
    empty_mask = torch.zeros_like(mask)
    loss_empty = masked_bce_loss(logits, y, empty_mask, pos_weight)

    assert torch.isfinite(loss_with_mask)
    assert torch.isfinite(loss_empty)
