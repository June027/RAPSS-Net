from __future__ import annotations

import numpy as np


def reliability_bin_stats(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    empirical_rates = []
    confidence_means = []
    counts = []
    for i in range(int(n_bins)):
        lo, hi = bins[i], bins[i + 1]
        if i == int(n_bins) - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if not np.any(mask):
            empirical_rates.append(np.nan)
            confidence_means.append(np.nan)
            counts.append(0)
            continue
        empirical_rates.append(float(np.mean(y_true[mask])))
        confidence_means.append(float(np.mean(y_prob[mask])))
        counts.append(int(np.sum(mask)))
    return empirical_rates, confidence_means, counts


def expected_calibration_error(y_true, y_prob, n_bins=10):
    empirical_rates, confidence_means, counts = reliability_bin_stats(
        y_true, y_prob, n_bins=n_bins
    )
    total = max(len(np.asarray(y_true)), 1)
    ece = 0.0
    for empirical_rate, confidence_mean, count in zip(
        empirical_rates, confidence_means, counts
    ):
        if count <= 0:
            continue
        ece += (count / total) * abs(empirical_rate - confidence_mean)
    return float(ece)
