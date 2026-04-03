import numpy as np

from src.evaluation.metrics import compute_optimal_threshold


def test_compute_optimal_threshold_returns_probability_cutoff():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.05, 0.10, 0.30, 0.35, 0.80, 0.95])

    threshold = compute_optimal_threshold(y_true, y_prob)

    assert 0.0 <= threshold <= 1.0
    assert threshold in y_prob
