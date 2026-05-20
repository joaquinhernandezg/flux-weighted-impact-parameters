"""Utilities for flux-weighted impact-parameter analyses in lensed systems."""

from .azimuthal_angle_map import calculate_azimuthal_angle, make_azimuthal_angle_map
from .binned_pixel_stats import get_binned_pixel_stats
from .correlation import (
    compute_normalized_correlations_matrix,
    compute_pixel_correlation_matrix,
    effective_number_of_independent_pixels,
    overlap_coefficients,
    covariance_matrix,
    correlation_matrix,
    geometrical_covariance,
    geometrical_covariance_matrix,
    covariance_matrix_fast,

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

from .simulation import exponential_radial_profile, get_observed_value
from .fitting import log_likelihood
from .rebinning import (downscale_highres_image, upsample_low_res_mask, 
                        get_extent, get_extent_in_other_frame, make_pixel_geometric_mask)

from .plot_utils import (
    plot_HST_and_MUSE_cutouts,
    plot_HST_original_convolved_rebinned_and_mask,
    plot_spaxels_on_highres,
    plot_spaxels_on_lowres,
    plot_example_contribution_map_for_binned_pixel,
    plot_hst_binned_pixels_contribution_map_and_impact_parameter_distribution,
    plot_displacement_maps,
)

from .utils import make_summary_table_spaxels



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
    "log_likelihood",
    "exponential_radial_profile",
    "get_observed_value",
    "covariance_matrix",
    "correlation_matrix",
    "geometrical_covariance",
    "geometrical_covariance_matrix",
    "covariance_matrix_fast",
    "get_extent",
    "get_extent_in_other_frame",
    "downscale_highres_image",
    "upsample_low_res_mask",
    "make_pixel_geometric_mask",
    "plot_HST_and_MUSE_cutouts",
    "plot_HST_original_convolved_rebinned_and_mask",
    "plot_spaxels_on_highres",
    "plot_spaxels_on_lowres",
    "plot_example_contribution_map_for_binned_pixel",
    "make_summary_table_spaxels",
    "plot_hst_binned_pixels_contribution_map_and_impact_parameter_distribution",
    "plot_displacement_maps",
]
