"""Evaluation metrics for early, rare-event sepsis prediction. Accuracy is
deliberately not implemented here — see CLAUDE.md §2: it's not an acceptable
primary metric at ~1.8% prevalence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from src import config


def auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_prob))


def auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_prob))


def sensitivity_specificity(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = config.ALERT_THRESHOLD
) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return {
        "threshold": threshold,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def early_prediction_gain(
    df: pd.DataFrame,
    prob_col: str = "y_prob",
    threshold: float = config.ALERT_THRESHOLD,
    patient_col: str = "patient_id",
    hour_col: str = "ICULOS",
    label_col: str = config.LABEL_COLUMN,
) -> dict[str, float]:
    """Mean hours between the first alert crossing `threshold` and actual
    onset (first hour with label==1), averaged across true-positive patients
    (patients who developed sepsis AND were alerted at or before onset).

    A patient counts toward the average only if the alert fires at or before
    the onset hour — an alert that only fires after onset contributes 0 gain
    and is excluded from the "gain" average but tracked as a late detection.
    """
    gains: list[float] = []
    late_detections = 0
    missed = 0

    for patient_id, group in df.groupby(patient_col):
        group = group.sort_values(hour_col)
        if group[label_col].max() == 0:
            continue  # never septic, not part of this metric

        onset_hour = group.loc[group[label_col] == 1, hour_col].min()
        alerts = group.loc[group[prob_col] >= threshold, hour_col]

        if alerts.empty:
            missed += 1
            continue

        first_alert_hour = alerts.min()
        gain = onset_hour - first_alert_hour
        if gain >= 0:
            gains.append(float(gain))
        else:
            late_detections += 1

    return {
        "mean_early_gain_hours": float(np.mean(gains)) if gains else 0.0,
        "n_true_positive_patients": len(gains),
        "n_late_detections": late_detections,
        "n_missed_patients": missed,
    }


def _simplified_utility_at_offset(hours_from_onset: float) -> float:
    """Simplified version of the PhysioNet 2019 Challenge utility function
    (docs/requirements_spec.md §10). Piecewise-linear reward for an alert
    fired `hours_from_onset` hours relative to actual sepsis onset (negative
    = before onset). This is a documented simplification of the official
    U(s,t) formula, not a reimplementation of it.
    """
    early = config.UTILITY_EARLY_WINDOW_HOURS  # e.g. 12
    late = config.UTILITY_LATE_WINDOW_HOURS  # e.g. 3

    if hours_from_onset < -early:
        return 0.0
    if -early <= hours_from_onset < -6:
        # ramp 0 -> 1 as we approach 6h-before-onset (the clinically ideal window)
        return (hours_from_onset + early) / (early - 6)
    if -6 <= hours_from_onset <= late:
        return 1.0
    if late < hours_from_onset <= late + 6:
        return 1.0 - (hours_from_onset - late) / 6
    return 0.0


def utility_score(
    df: pd.DataFrame,
    prob_col: str = "y_prob",
    threshold: float = config.ALERT_THRESHOLD,
    patient_col: str = "patient_id",
    hour_col: str = "ICULOS",
    label_col: str = config.LABEL_COLUMN,
    false_alarm_penalty: float = 0.05,
) -> float:
    """Simplified clinical utility score, averaged per patient then across
    patients. Positive alerts near the ideal early window score close to 1;
    alerts far too early/late decay toward 0; false alarms on never-septic
    patients incur a small constant penalty.
    """
    per_patient_scores: list[float] = []

    for patient_id, group in df.groupby(patient_col):
        group = group.sort_values(hour_col)
        alerts = group[group[prob_col] >= threshold]

        if group[label_col].max() == 1:
            onset_hour = group.loc[group[label_col] == 1, hour_col].min()
            if alerts.empty:
                per_patient_scores.append(0.0)
                continue
            # score every alert relative to onset, take the best (earliest
            # well-timed) alert as the patient's score
            offsets = alerts[hour_col] - onset_hour
            scores = [_simplified_utility_at_offset(o) for o in offsets]
            per_patient_scores.append(max(scores))
        else:
            n_alerts = len(alerts)
            per_patient_scores.append(-false_alarm_penalty * n_alerts / max(1, len(group)))

    return float(np.mean(per_patient_scores)) if per_patient_scores else 0.0


def full_report(
    df: pd.DataFrame,
    prob_col: str = "y_prob",
    threshold: float = config.ALERT_THRESHOLD,
) -> dict:
    """Everything required by CLAUDE.md §6's evaluation contract, in one call."""
    y_true = df[config.LABEL_COLUMN].to_numpy()
    y_prob = df[prob_col].to_numpy()

    report = {
        "auroc": auroc(y_true, y_prob),
        "auprc": auprc(y_true, y_prob),
        **sensitivity_specificity(y_true, y_prob, threshold),
        **early_prediction_gain(df, prob_col=prob_col, threshold=threshold),
        "utility_score": utility_score(df, prob_col=prob_col, threshold=threshold),
    }
    return report
