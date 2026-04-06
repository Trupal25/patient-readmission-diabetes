import numpy as np

from src.evaluation.metrics import compute_optimal_threshold


def test_compute_optimal_threshold_returns_probability_cutoff():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.05, 0.10, 0.30, 0.35, 0.80, 0.95])

    threshold = compute_optimal_threshold(y_true, y_prob)

    assert np.isfinite(threshold)
    assert 0.0 <= threshold <= 1.0
    assert threshold == 0.35


def test_compute_optimal_threshold_accounts_for_absolute_false_positive_burden():
    y_true = np.array([0, 0, 0, 0, 0, 0, 1])
    y_prob = np.array([0.90, 0.80, 0.70, 0.60, 0.55, 0.51, 0.50])

    threshold = compute_optimal_threshold(y_true, y_prob)

    assert threshold == 1.0
