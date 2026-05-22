# flux-weighted-impact-parameters

Python utilities for flux-weighted impact parameter and azimuthal angle analyses in gravitationally lensed systems. Designed for IFU spectroscopy (MUSE, KCWI) cross-matched with high-resolution imaging (HST), where each low-resolution spaxel sees a mix of source-plane positions.

The core idea: rather than assigning a single impact parameter to a spaxel, the code weights each source-plane position by how much flux it contributes to that spaxel after PSF convolution and spatial resampling.

## Repository layout

```
flux-weighted-impact-parameters/
├── pyproject.toml
└── src/
    └── flux_weighted_impact_parameters/
        ├── lensing.py                    # lensing matrix scaling, delensing, magnification, critical lines
        ├── impact_parameter_map.py       # source-plane impact parameter maps in kpc
        ├── azimuthal_angle_map.py        # azimuthal angle maps relative to galaxy major axis
        ├── flux_contribution_maps.py     # per-spaxel flux contribution kernels
        ├── impact_parameter_distribution.py  # flux-weighted stats per spaxel
        ├── binned_pixel_stats.py         # orchestrates all per-spaxel statistics
        ├── correlation.py                # spaxel correlation and covariance matrices
        ├── segmentation.py               # arc detection and masking
        ├── rebinning.py                  # image downscaling and pixel-grid utilities
        ├── simulation.py                 # exponential radial profile and forward model
        ├── fitting.py                    # log-likelihood with correlated errors
        ├── stats_utils.py                # weighted mean, std, percentile
        ├── plot_utils.py                 # visualization helpers
        └── utils.py                      # summary table export
```

## Installation

Clone and install in editable mode:

```bash
pip install -e .
```

**Dependencies**: `astropy`, `matplotlib`, `mpdaf`, `numpy`, `photutils`, `scikit-image`, `scipy`.

## Workflow overview

A complete end-to-end example is in [`notebooks/example_notebook_on_SGASJ1226.ipynb`](notebooks/example_notebook_on_SGASJ1226.ipynb). The notebook covers all steps below using real data.

### 1. Prepare lensing products

Scale deflection maps from the normalization redshift to your lens/source configuration using `scale_lensing_matrix`. Compute magnification, convergence, and shear from the deflection fields with `compute_magnification_map_from_deflections` and `compute_kappa_shear_map_from_deflections` when no pre-computed maps are available.

### 2. Segment the arc

Use `make_segmentation_map_and_mask` to run source detection on the HST cutout and produce a binary arc mask. A first interactive pass visualizes the segmentation labels; subsequent runs can be driven non-interactively by passing the selected label list directly.

### 3. Build source-plane maps

`make_impact_parameter_map` and `make_azimuthal_angle_map` delens every HST pixel center through the deflection fields and compute, for each pixel, the transverse impact parameter in kpc and the azimuthal angle relative to the galaxy major axis. Both maps are written as FITS files and share the WCS of the input HST image.

The azimuthal angle is defined such that 0° is along the galaxy major axis and increases counterclockwise (east of north).

### 4. Rebin to IFU grid

`downscale_highres_image` rebins the HST cutout to the IFU pixel scale while conserving flux and keeping WCS alignment consistent between the two grids.

### 5. Compute per-spaxel statistics

`get_binned_pixel_stats` iterates over all masked spaxels. For each one it computes the flux contribution kernel (which HST pixels contribute and with what weight after PSF convolution), the flux-weighted impact parameter and azimuthal angle distributions (mean, std, percentiles), the local magnification, and the distance to the critical line. Results are returned as a dict keyed by `(x_pix, y_pix)` and can be exported to an astropy Table with `make_summary_table_spaxels`.


## Module reference

| Module | Key functions |
|---|---|
| `lensing` | `scale_lensing_matrix`, `make_delens`, `compute_magnification_map_from_deflections`, `compute_kappa_shear_map_from_deflections`, `get_critical_line_mask`, `distance_to_critical_line_map` |
| `impact_parameter_map` | `make_impact_parameter_map`, `impact_parameter_kpc` |
| `azimuthal_angle_map` | `make_azimuthal_angle_map`, `calculate_azimuthal_angle` |
| `flux_contribution_maps` | `compute_input_contribution_map_for_binned_pixel`, `compute_arc_sky_flux_contributions` |
| `impact_parameter_distribution` | `impact_parameter_stats_for_binned_pixel`, `azimuthal_angle_stats_for_binned_pixel` |
| `binned_pixel_stats` | `get_binned_pixel_stats` |
| `correlation` | `compute_pixel_correlation_matrix`, `geometrical_covariance_matrix`, `covariance_matrix_fast`, `correlation_matrix`, `overlap_coefficients`, `effective_number_of_independent_pixels` |
| `segmentation` | `make_segmentation_map_and_mask`, `build_segmentation_map`, `keep_sources_within_radius`, `save_segmentation_mask_as_fits` |
| `rebinning` | `downscale_highres_image`, `upsample_low_res_mask`, `make_pixel_geometric_mask`, `get_extent`, `get_extent_in_other_frame` |
| `simulation` | `exponential_radial_profile`, `get_observed_value` |
| `fitting` | `log_likelihood` |
| `stats_utils` | `weighted_mean`, `weighted_std`, `weighted_percentile` |
| `utils` | `make_summary_table_spaxels` |
| `plot_utils` | `plot_HST_and_MUSE_cutouts`, `plot_input_contribution_map_for_binned_pixel`, `plot_spaxels_on_highres`, `plot_spaxels_on_lowres`, `plot_displacement_maps`, and more |

## Notes on conventions

- Deflection fields are expected in arcseconds, following the LENSTOOL convention: positive α_x deflects toward decreasing RA, positive α_y deflects toward increasing Dec.
- Lensing matrix scaling assumes single-plane lensing. Do not apply `scale_lensing_matrix` to deflection fields from multi-plane codes.
- Cosmology defaults to a flat ΛCDM with H₀ = 70 km/s/Mpc, Ω_m = 0.3. Pass a custom `astropy.cosmology` instance to override.
- Large FITS files and figures are excluded by `.gitignore`.
