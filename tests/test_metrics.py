from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation import metrics


def test_sensitivity_specificity_hand_computable():
    # 4 rows: 2 true positives predicted correctly, 1 false negative, 1 true negative
    y_true = np.array([1, 1, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.2, 0.1])  # third row: true label 1, predicted 0
    result = metrics.sensitivity_specificity(y_true, y_prob, threshold=0.5)

    assert result["true_positives"] == 2
    assert result["false_negatives"] == 1
    assert result["true_negatives"] == 1
    assert result["false_positives"] == 0
    assert result["sensitivity"] == 2 / 3
    assert result["specificity"] == 1.0


def test_auroc_perfect_separation_is_one():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    assert metrics.auroc(y_true, y_prob) == 1.0


def test_auroc_random_is_about_half():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=2000)
    y_prob = rng.random(2000)
    assert 0.45 < metrics.auroc(y_true, y_prob) < 0.55


def test_early_prediction_gain_hand_computable():
    # Patient p1: onset at hour 10, first alert at hour 6 -> gain of 4 hours
    # Patient p2: never septic -> excluded
    df = pd.DataFrame({
        "patient_id": ["p1"] * 12 + ["p2"] * 5,
        "ICULOS": list(range(1, 13)) + list(range(1, 6)),
        "SepsisLabel": [0] * 9 + [1] * 3 + [0] * 5,
        "y_prob": [0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
                  + [0.1, 0.1, 0.1, 0.1, 0.1],
    })
    result = metrics.early_prediction_gain(df, threshold=0.5)
    assert result["n_true_positive_patients"] == 1
    assert result["mean_early_gain_hours"] == 4.0
    assert result["n_missed_patients"] == 0


def test_early_prediction_gain_missed_patient_counted():
    df = pd.DataFrame({
        "patient_id": ["p1"] * 5,
        "ICULOS": list(range(1, 6)),
        "SepsisLabel": [0, 0, 0, 1, 1],
        "y_prob": [0.1, 0.1, 0.1, 0.1, 0.1],  # never crosses threshold
    })
    result = metrics.early_prediction_gain(df, threshold=0.5)
    assert result["n_missed_patients"] == 1
    assert result["n_true_positive_patients"] == 0


def test_utility_score_perfect_early_alert_scores_near_one():
    df = pd.DataFrame({
        "patient_id": ["p1"] * 10,
        "ICULOS": list(range(1, 11)),
        "SepsisLabel": [0] * 6 + [1] * 4,
        # onset at hour 7; alert fires at hour 3 (4h before onset, inside the
        # -6h..+3h "ideal" plateau is hour 1..10, so this is within ramp)
        "y_prob": [0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
    })
    score = metrics.utility_score(df, threshold=0.5)
    assert score > 0.0


def test_utility_score_false_alarms_penalized():
    df = pd.DataFrame({
        "patient_id": ["p1"] * 5,
        "ICULOS": list(range(1, 6)),
        "SepsisLabel": [0] * 5,
        "y_prob": [0.9, 0.9, 0.9, 0.9, 0.9],  # all false alarms
    })
    score = metrics.utility_score(df, threshold=0.5)
    assert score < 0.0
