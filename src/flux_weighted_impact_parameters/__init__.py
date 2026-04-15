"""Utilities for flux-weighted impact-parameter analyses in lensed systems."""

from .azimuthal_angle_map import calculate_azimuthal_angle, make_azimuthal_angle_map
from .binned_pixel_stats import get_binned_pixel_stats
from .correlation import (
    compute_normalized_correlations_matrix,
    compute_pixel_correlation_matrix,
    effective_number_of_independent_pixels,
    overlap_coefficients,
)
from .flux_contribution_maps import (
    compute_arc_sky_flux_contributions,
    compute_input_contribution_map_for_binned_pixel,
    plot_input_contribution_map_for_binned_pixel,
)
from .impact_parameter_distribution import (
    azimuthal_angle_stats_for_binned_pixel,
    impact_parameter_stats_for_binned_pixel,
    weighted_mean,
    weighted_percentile,
    weighted_std,
)
from .impact_parameter_map import impact_parameter_kpc, make_impact_parameter_map
from .lensing import (
    compute_kappa_shear_map_from_deflections,
    compute_magnification_map_from_deflections,
    distance_to_critical_line_map,
    get_critical_line_mask,
    make_delens,
    scale_lensing_matrix,
)
from .make_segmentation_mask import (
    make_segmentation_map_and_mask,
    save_segmentation_mask_as_fits,
)
from .segmentation_utils import (
    build_segmentation_map,
    keep_sources_within_radius,
    plot_segmentation_with_labels,
)

__all__ = [
    "azimuthal_angle_stats_for_binned_pixel",
    "build_segmentation_map",
    "calculate_azimuthal_angle",
    "compute_arc_sky_flux_contributions",
    "compute_input_contribution_map_for_binned_pixel",
    "compute_kappa_shear_map_from_deflections",
    "compute_magnification_map_from_deflections",
    "compute_normalized_correlations_matrix",
    "compute_pixel_correlation_matrix",
    "distance_to_critical_line_map",
    "effective_number_of_independent_pixels",
    "get_binned_pixel_stats",
    "get_critical_line_mask",
    "impact_parameter_kpc",
    "impact_parameter_stats_for_binned_pixel",
    "keep_sources_within_radius",
    "make_azimuthal_angle_map",
    "make_delens",
    "make_impact_parameter_map",
    "make_segmentation_map_and_mask",
    "overlap_coefficients",
    "plot_input_contribution_map_for_binned_pixel",
    "plot_segmentation_with_labels",
    "save_segmentation_mask_as_fits",
    "scale_lensing_matrix",
    "weighted_mean",
    "weighted_percentile",
    "weighted_std",
]
