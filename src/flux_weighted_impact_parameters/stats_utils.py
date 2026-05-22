import numpy as np


def weighted_percentile(values, weights, percentiles):
    """
    Weighted percentiles of a 1D distribution.

    Parameters
    ----------
    values : array-like
        Data values.
    weights : array-like
        Non-negative weights.
    percentiles : float or sequence
        Percentiles in [0, 100].

    Returns
    -------
    result : float or ndarray
        Weighted percentile value(s).
    """
    
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    percentiles = np.atleast_1d(percentiles).astype(float)

    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[good]
    weights = weights[good]

    if len(values) == 0:
        out = np.full(len(percentiles), np.nan)
        return out[0] if np.ndim(percentiles) == 0 else out

    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]

    cdf = np.cumsum(weights)
    cdf /= cdf[-1]

    out = np.interp(percentiles / 100.0, cdf, values)
    return out[0] if np.ndim(percentiles) == 0 else out


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(good):
        return np.nan
    return np.sum(weights[good] * values[good]) / np.sum(weights[good])


def weighted_std(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(good):
        return np.nan
    mu = weighted_mean(values[good], weights[good])
    var = np.sum(weights[good] * (values[good] - mu)**2) / np.sum(weights[good])
    return np.sqrt(var)
