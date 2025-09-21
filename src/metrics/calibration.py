"""
Probabilistic Metrics and Calibration Utilities

This module provides comprehensive tools for evaluating probabilistic forecasts,
including calibration, sharpness, and proper scoring rules.

All functions work with numpy arrays and return numpy values.
"""

import numpy as np
from scipy import stats
from typing import Tuple, Union
import warnings

def crps_from_samples(y: np.ndarray, samples: np.ndarray):
    """
    CRPS from predictive samples (for MC-Dropout/ensembles).
    """
    y = y[:, None]               
    S = samples.shape[1]
    term1 = np.mean(np.abs(samples - y), axis=1)           
    #V-statistic
    diffs = np.abs(samples[:, :, None] - samples[:, None, :])  
    term2 = 0.5 * np.mean(diffs, axis=(1,2))
    return np.mean(term1 - term2)

def combine_normal_moments(mus: np.ndarray, vars_: np.ndarray, axis=0):
    """
    Combine multiple Normal distributions via moment matching.
    """
    mu_bar = np.mean(mus, axis=axis)
    # total variance = mean(var) + var(mean)
    tot_var = np.mean(vars_, axis=axis) + np.var(mus, axis=axis, ddof=0)
    sigma_bar = np.sqrt(np.maximum(tot_var, 1e-12))
    return mu_bar, sigma_bar

def pit_from_samples(y: np.ndarray, samples: np.ndarray):
    """
    PIT values from predictive samples.
    """
    # empirical CDF at y
    return np.mean(samples <= y[:, None], axis=1)

def gaussian_nll(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    """
    Gaussian Negative Log-Likelihood (proper scoring rule).
    """
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Gaussian NLL
    nll = 0.5 * (np.log(2 * np.pi) + 2 * np.log(sigma) + (y - mu)**2 / sigma**2)
    return np.mean(nll)

def gaussian_crps(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    """
    Closed-form CRPS for Gaussian distributions.
    """
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Standardized errors
    z = (y - mu) / sigma
    
    phi_z = stats.norm.cdf(z)
    phi_z_std = stats.norm.pdf(z)
    
    crps = sigma * (z * (2 * phi_z - 1) + 2 * phi_z_std - 1/np.sqrt(np.pi))
    return np.mean(crps)

def prediction_intervals(mu: np.ndarray, sigma: np.ndarray, alpha: Union[float, list] = 0.1):
    """
    Compute prediction intervals for given confidence level(s).
    """
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    if isinstance(alpha, (int, float)):
        # Single alpha
        z_alpha = stats.norm.ppf(1 - alpha/2)
        lower = mu - z_alpha * sigma
        upper = mu + z_alpha * sigma
        return lower, upper
    else:
        # Multiple alphas
        result = {}
        for a in alpha:
            z_alpha = stats.norm.ppf(1 - a/2)
            lower = mu - z_alpha * sigma
            upper = mu + z_alpha * sigma
            result[a] = (lower, upper)
        return result

def interval_coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray):
    """
    Compute interval coverage and mean width (sharpness).
    """
    # Coverage: fraction of observations within interval
    covered = (y >= lower) & (y <= upper)
    coverage = np.mean(covered) * 100
    
    # Sharpness: mean interval width
    mean_width = np.mean(upper - lower)
    
    return coverage, mean_width

def pit_values(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    """
    Probability Integral Transform (PIT) values for calibration assessment.
    """
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # PIT = F(y) where F is the predicted CDF
    pit = stats.norm.cdf(y, loc=mu, scale=sigma)
    return pit

def event_probability(mu: np.ndarray, sigma: np.ndarray, threshold: float, side: str = '>'):
    """
    Compute probability of exceeding/falling below threshold using Normal CDF.
    """
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    if side == '>':
        # P(Y > threshold) = 1 - F(threshold)
        prob = 1 - stats.norm.cdf(threshold, loc=mu, scale=sigma)
    elif side == '<':
        # P(Y < threshold) = F(threshold)
        prob = stats.norm.cdf(threshold, loc=mu, scale=sigma)
    else:
        raise ValueError("side must be '>' or '<'")
    
    return prob

def reliability_curve(p: np.ndarray, y_event: np.ndarray, n_bins: int = 10):
    """
    Compute reliability curve for binary event forecasts.
    """
    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(p, bin_edges) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    p_bin = np.zeros(n_bins)          # mean forecast prob in bin
    obs_bin = np.zeros(n_bins)        # observed frequency in bin
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        m = (bin_ids == i)
        if m.any():
            counts[i] = m.sum()
            p_bin[i]  = p[m].mean()
            obs_bin[i]= y_event[m].mean()

    return p_bin, obs_bin, counts  # <-- return p_bin, not bin centers

def brier_score(p: np.ndarray, y_event: np.ndarray):
    """
    Brier Score for binary event forecasts.
    """
    return np.mean((p - y_event)**2)

def brier_score_decomposition(p: np.ndarray, y_event: np.ndarray, n_bins: int = 10):
    """
    Decompose Brier Score into reliability, resolution, and uncertainty.
    """
    # Overall Brier Score
    bs = brier_score(p, y_event)
    
    # Get reliability curve data
    p_bin, obs_bin, counts = reliability_curve(p, y_event, n_bins)
    
    # Overall event frequency
    p_clim = np.mean(y_event)
    
    # Reliability
    reliability = np.sum(counts * (p_bin - obs_bin)**2) / len(p)
    
    # Resolution
    resolution = np.sum(counts * (obs_bin - p_clim)**2) / len(p)
    
    # Uncertainty
    uncertainty = p_clim * (1 - p_clim)
    
    return bs, reliability, resolution, uncertainty

def calibration_summary(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, 
                       thresholds: list = None, intervals: tuple = (0.5, 0.9)):
    """
    Comprehensive calibration summary for probabilistic forecasts.
    """
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    if thresholds is None:
        thresholds = [np.percentile(y, p) for p in (25, 50, 75)]
    
    # Basic scores
    nll = gaussian_nll(y, mu, sigma)
    crps = gaussian_crps(y, mu, sigma)
    
    # Multiple prediction intervals
    cov = {}; width = {}
    for q in intervals:
        lo, hi = prediction_intervals(mu, sigma, alpha=1-q)
        c, w = interval_coverage(y, lo, hi)
        cov[int(q*100)] = c; width[int(q*100)] = w
    
    # PIT values
    pit = pit_values(y, mu, sigma)
    
    # Event probabilities (mean values for summary)
    ev = {f'P(Y>{thr:.1f})_mean': event_probability(mu, sigma, thr).mean() for thr in thresholds}
    
    return {
        'nll': nll, 'crps': crps,
        'coverage': cov, 'width': width,
        'pit_mean': np.mean(pit), 'pit_std': np.std(pit),
        'event_prob_means': ev
    }

def plot_reliability_curve(p: np.ndarray, y_event: np.ndarray, n_bins: int = 10, 
                          ax=None, title: str = "Reliability Curve"):
    """
    Plot reliability curve for binary event forecasts.
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    p_bin, obs_bin, counts = reliability_curve(p, y_event, n_bins)
    
    # Plot reliability curve
    ax.plot(p_bin, obs_bin, 'o-', linewidth=2, label='Reliability')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Perfect calibration')
    
    # Add sample size annotations
    for x, y, c in zip(p_bin, obs_bin, counts):
        if c > 0:
            ax.annotate(f'n={int(c)}', (x, y), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8, alpha=0.7)
    
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Observed Frequency')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

def plot_pit_histogram(pit: np.ndarray, ax=None, title: str = "PIT Histogram"):
    """
    Plot PIT histogram for calibration assessment.
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    
    # Plot histogram
    ax.hist(pit, bins=20, density=True, alpha=0.7, edgecolor='black')
    ax.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Perfect calibration (uniform)')
    
    ax.set_xlabel('PIT Value')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
