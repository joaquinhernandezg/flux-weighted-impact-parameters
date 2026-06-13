"""
Monte Carlo Uncertainty Extraction from ODR Fits
================================================
Wraps the ODR → MC sampling → metrics pipeline into reusable functions.
Generates comparison tables for impact param & scatter distributions.

Usage:
  - Fit with ODR (your existing code)
  - Pass (pop, pcov) to MCUncertaintyExtractor
  - Call .extract_metrics() to get table row
  - Aggregate across perturbations/methods
"""

import numpy as np
import pandas as pd
from scipy.odr import ODR, Model, RealData
import matplotlib.pyplot as plt


class MCUncertaintyExtractor:
    """
    Extracts uncertainties from ODR fit via MC sampling of covariance matrix.
    
    Parameters:
    -----------
    odr_result : scipy.odr.ODROutput
        Output from ODR.run()
    profile_func : callable
        Evaluation function: profile_func(b, scale_radius, normalization)
    name : str
        Label for this fit (e.g., 'Spaxel-centre', 'Flux-weighted')
    n_mc : int
        Number of MC samples (default 3000)
    """
    
    def __init__(self, odr_result, profile_func, name, n_mc=3000):
        self.pop = odr_result.beta
        self.pcov = odr_result.cov_beta
        self.profile_func = profile_func
        self.name = name
        self.n_mc = n_mc
        self._samples = None
        self._eval_points = None
        
    def sample_posterior(self, random_seed=None):
        """Draw MC samples from the ODR posterior (multivariate normal)."""
        if random_seed is not None:
            np.random.seed(random_seed)
        
        samples = np.random.multivariate_normal(
            mean=self.pop, 
            cov=self.pcov, 
            size=self.n_mc
        )
        # Keep only physically meaningful (positive) parameters
        mask = (samples[:, 0] > 0) & (samples[:, 1] > 0)
        self._samples = samples[mask]
        self.n_valid = len(self._samples)
        return self._samples
    
    def evaluate_profile(self, b_eval=None):
        """
        Evaluate the profile at given impact parameters.
        Returns: (b_eval, y_samples, y16, y50, y84)
        """
        if self._samples is None:
            self.sample_posterior()
        
        if b_eval is None:
            b_eval = np.linspace(0, 35, 100)
        
        y_samples = np.array([
            np.log10(self.profile_func(b_eval, p[0], 10**p[1])) 
            for p in self._samples
        ])
        
        y16, y50, y84 = np.percentile(y_samples, [16, 50, 84], axis=0)
        
        return b_eval, y_samples, y16, y50, y84
    
    def fit_params_percentiles(self):
        """
        Extract percentile ranges for fitted parameters.
        Returns: dict with scale_radius and normalization stats
        """
        if self._samples is None:
            self.sample_posterior()
        
        scale_radius_samples = self._samples[:, 0]
        norm_samples = 10**self._samples[:, 1]  # Back to linear space
        
        return {
            'scale_radius_median': np.median(scale_radius_samples),
            'scale_radius_p16': np.percentile(scale_radius_samples, 16),
            'scale_radius_p84': np.percentile(scale_radius_samples, 84),
            'norm_median': np.median(norm_samples),
            'norm_p16': np.percentile(norm_samples, 16),
            'norm_p84': np.percentile(norm_samples, 84),
        }
    
    def _residuals(self, b_list, N_model_list):
        """Compute residuals for each MC sample at given (b, N) points."""
        residuals_per_sample = []
        for p in self._samples:
            model_vals = self.profile_func(b_list, p[0], 10**p[1])
            resid = np.abs(np.log10(model_vals) - np.log10(N_model_list))
            residuals_per_sample.append(resid)
        return np.array(residuals_per_sample)  # shape: (n_samples, n_points)
    
    def scatter_distribution(self, b_list, N_model_list):
        """
        Calculate scatter (mean |residual|) for each MC sample.
        Returns: scatter_array, median, p16, p84
        """
        residuals_all = self._residuals(b_list, N_model_list)
        scatter_per_sample = np.mean(residuals_all, axis=1)  # mean over points
        
        return {
            'scatter_samples': scatter_per_sample,
            'scatter_median': np.median(scatter_per_sample),
            'scatter_p16': np.percentile(scatter_per_sample, 16),
            'scatter_p84': np.percentile(scatter_per_sample, 84),
            'scatter_mean': np.mean(scatter_per_sample),
            'scatter_std': np.std(scatter_per_sample),
        }
    
    def impact_param_bias(self, b_true, b_meas_list):
        """
        Compute bias in impact parameter measurements.
        b_true: true value (scalar)
        b_meas_list: array of measured values (across many trials/spaxels)
        
        Returns: dict with bias stats
        """
        bias = b_meas_list - b_true
        return {
            'bias_median': np.median(bias),
            'bias_mean': np.mean(bias),
            'bias_std': np.std(bias),
            'bias_p16': np.percentile(bias, 16),
            'bias_p84': np.percentile(bias, 84),
        }


def create_comparison_table(
    extractors_dict,
    b_measured_dict,
    b_true,
    N_model_dict,
    b_list_dict,
):
    """
    Build a comparison table across multiple fits/perturbations.
    
    Parameters:
    -----------
    extractors_dict : dict
        {'method_name': MCUncertaintyExtractor, ...}
    b_measured_dict : dict
        {'method_name': array of measured impact params, ...}
    b_true : float
        True impact parameter value
    N_model_dict : dict
        {'method_name': array of column densities at each b, ...}
    b_list_dict : dict
        {'method_name': array of impact parameters, ...}
    
    Returns:
    --------
    pd.DataFrame
    """
    rows = []
    
    for name, extractor in extractors_dict.items():
        # Ensure samples exist
        if extractor._samples is None:
            extractor.sample_posterior()
        
        # Impact param bias
        bias_stats = extractor.impact_param_bias(b_true, b_measured_dict[name])
        
        # Scatter around best-fit profile
        scatter_stats = extractor.scatter_distribution(
            b_list_dict[name], 
            N_model_dict[name]
        )
        
        # Param uncertainties
        param_stats = extractor.fit_params_percentiles()
        
        row = {
            'Method': name,
            'N_MC_valid': extractor.n_valid,
            '─────────': '─────────',
            'Bias_median_kpc': f"{bias_stats['bias_median']:+.4f}",
            'Bias_std_kpc': f"{bias_stats['bias_std']:.4f}",
            'Bias_p16_p84': f"[{bias_stats['bias_p16']:+.4f}, {bias_stats['bias_p84']:+.4f}]",
            '──────────────': '──────────────',
            'Scatter_dex': f"{scatter_stats['scatter_median']:.4f}",
            'Scatter_p16_p84': f"[{scatter_stats['scatter_p16']:.4f}, {scatter_stats['scatter_p84']:.4f}]",
            '───────────────': '───────────────',
            'scale_radius_kpc': f"{param_stats['scale_radius_median']:.4f} (+{param_stats['scale_radius_p84'] - param_stats['scale_radius_median']:.4f} / -{param_stats['scale_radius_median'] - param_stats['scale_radius_p16']:.4f})",
            'normalization': f"{param_stats['norm_median']:.3e} (+{param_stats['norm_p84'] - param_stats['norm_median']:.3e} / -{param_stats['norm_median'] - param_stats['norm_p16']:.3e})",
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def summary_table(extractors_dict):
    """
    Minimal summary: method name, N_MC, scatter, scale_radius
    """
    rows = []
    for name, ext in extractors_dict.items():
        if ext._samples is None:
            ext.sample_posterior()
        
        param_stats = ext.fit_params_percentiles()
        rows.append({
            'Method': name,
            'N_MC': ext.n_valid,
            'scale_radius (kpc)': f"{param_stats['scale_radius_median']:.3f} ± {(param_stats['scale_radius_p84'] - param_stats['scale_radius_p16']) / 2:.3f}",
        })
    
    return pd.DataFrame(rows)


# ============================================================================
# PLOTTING UTILITIES
# ============================================================================

def plot_profile_comparison(
    extractors_dict,
    odr_results_dict,
    N_data_dict,
    b_data_dict,
    b_err_dict,
    colors=None,
    b_max=35,
    figsize=(10, 5),
):
    """
    Plot fitted profiles with MC confidence bands for multiple methods.
    """
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(extractors_dict)))
    
    b_fit = np.linspace(0, b_max, 100)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for (name, extractor), color in zip(extractors_dict.items(), colors):
        # Profile evaluations
        _, _, y16, y50, y84 = extractor.evaluate_profile(b_fit)
        
        ax.plot(b_fit, y50, color=color, linestyle='--', linewidth=2, label=name)
        ax.fill_between(b_fit, y16, y84, color=color, alpha=0.3)
        
        # Data points
        if name in N_data_dict:
            ax.errorbar(
                b_data_dict[name], np.log10(N_data_dict[name]),
                xerr=b_err_dict[name],
                fmt='o', color=color, markersize=4, elinewidth=0.5,
                alpha=0.6
            )
    
    ax.set_xlabel('Impact Parameter [kpc]', fontsize=11)
    ax.set_ylabel(r'log(N / cm$^{-2}$)', fontsize=11)
    ax.set_xlim(0, b_max)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    return fig, ax
