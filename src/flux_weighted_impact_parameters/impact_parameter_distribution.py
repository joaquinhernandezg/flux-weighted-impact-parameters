import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from .flux_contribution_maps import compute_input_contribution_map_for_binned_pixel


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


def impact_parameter_stats_for_binned_pixel(
    highres_image,
    lowres_image,
    impact_parameter_map,
    highres_psf_FWHM,
    lowres_psf_FWHM,
    x_pix_lowres,
    y_pix_lowres,
    arc_mask_highres=None,
    percentiles=(5, 16, 25, 50, 75, 84, 95),
    normalize_weights=True
):
    """
    Compute the flux-contribution-weighted impact-parameter distribution
    for one binned spaxel.

    Parameters
    ----------
    highres_image : mpdaf.obj.Image-like
        High-resolution image on the same pixel grid as impact_parameter_map.
    lowres_image : mpdaf.obj.Image-like
        Low-resolution image.
    impact_parameter_map : 2D ndarray
        Impact parameter map on the high-resolution grid. Must match highres_image.data shape.
    highres_psf_FWHM : float
        Gaussian FWHM of the high-resolution PSF in arcsec.
    lowres_psf_FWHM : float
        Gaussian FWHM of the low-resolution PSF in arcsec.
    x_pix_lowres, y_pix_lowres : int
        Coordinates of the selected binned spaxel in the low-resolution grid.
    arc_mask_highres : 2D bool/int array or None
        Optional arc mask on HST grid.
    percentiles : tuple
        Percentiles to compute.
    normalize_weights : bool
        If True, use normalized contribution map as weights.

    Returns
    -------
    result : dict
        Weighted stats and arrays.
    """
    contrib_map, weighted_flux_map, total_flux, sens_map = compute_input_contribution_map_for_binned_pixel(
        highres_image=highres_image,
        lowres_image=lowres_image,
        highres_psf_FWHM=highres_psf_FWHM,
        lowres_psf_FWHM=lowres_psf_FWHM,
        x_pix_lowres=x_pix_lowres,
        y_pix_lowres=y_pix_lowres,
        arc_mask_highres=arc_mask_highres,
        normalize=normalize_weights
    )

    weights_map = contrib_map if normalize_weights else weighted_flux_map

    if impact_parameter_map.shape != weights_map.shape:
        raise ValueError("impact_parameter_map and hst_cutout/contribution map must have the same shape")

    values = np.asarray(impact_parameter_map, dtype=float).ravel()
    weights = np.asarray(weights_map, dtype=float).ravel()

    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[good]
    weights = weights[good]

    # also calculate the impact parameter belonging to the centre in the low resolution image
    # calculate the vertices of the binned pixel in the high resolution grid
    # calculate the impact parameter at the center of the binned pixel in the low resolution image
    # and take the error as the range of impact parameters within the binned pixel corners in the high resolution grid
    x_corners = [x_pix_lowres-0.5, x_pix_lowres-0.5, x_pix_lowres+0.5, x_pix_lowres+0.5]
    y_corners = [y_pix_lowres-0.5, y_pix_lowres+0.5, y_pix_lowres-0.5, y_pix_lowres+0.5]

    ra_centre, dec_centre = lowres_image.wcs.wcs.wcs_pix2world(x_pix_lowres, y_pix_lowres, 0)
    x_centre_highres, y_centre_highres = highres_image.wcs.wcs.wcs_world2pix(ra_centre, dec_centre, 0)
    impact_parameter_centre = impact_parameter_map[y_centre_highres.astype(int), x_centre_highres.astype(int)]

    impact_parameter_corners = []
    for x_corner, y_corner in zip(x_corners, y_corners):
        ra_corner, dec_corner = lowres_image.wcs.wcs.wcs_pix2world(x_corner, y_corner, 0)
        x_corner_highres, y_corner_highres = highres_image.wcs.wcs.wcs_world2pix(ra_corner, dec_corner, 0)
        impact_parameter_corners.append(impact_parameter_map[y_corner_highres.astype(int), x_corner_highres.astype(int)])

    impact_parameter_centre_error = np.ptp(impact_parameter_corners)/2


    psf_FWHM_pix_highres = highres_psf_FWHM / highres_image.wcs.wcs.proj_plane_pixel_scales()[0].to_value('arcsec')
    # use the PSF radius to define a minimum error on the impact parameter centre, to account for the fact that the flux is not coming from a single point but is convolved with the PSF
    psf_radius_pix_highres = psf_FWHM_pix_highres / 2
    # mask a mask in the impact parameter map around the centre with a radius of the PSF to account for the fact that the flux is convolved with the PSF and not coming from a single point
    yy, xx = np.mgrid[0:impact_parameter_map.shape[0], 0:impact_parameter_map.shape[1]]
    r = np.sqrt((xx - x_centre_highres)**2 + (yy - y_centre_highres)**2)
    impact_parameter_error_psf = np.ptp(impact_parameter_map[r <= psf_radius_pix_highres])

    dx, dy = highres_image.wcs.wcs.proj_plane_pixel_scales()[0].to_value('arcsec'), highres_image.wcs.wcs.proj_plane_pixel_scales()[1].to_value('arcsec')
    df_dy, df_dx = np.gradient(impact_parameter_map.data, dy, dx)
    grad_mag = np.sqrt(df_dx**2 + df_dy**2)
    weighted_grad_mag = np.sum(contrib_map * grad_mag) / np.sum(contrib_map)


    if len(values) == 0:
        return {
            "x_pix_lowres": x_pix_lowres,
            "y_pix_lowres": y_pix_lowres,
            "total_flux": total_flux,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "percentiles": {p: np.nan for p in percentiles},
            "values": values,
            "weights": weights,
            "weights_map": weights_map,
            'impact_parameter_centre': impact_parameter_centre,
            'impact_parameter_centre_error': impact_parameter_centre_error,
            'impact_parameter_error_psf': impact_parameter_error_psf,
            'local_gradient_magnitude': weighted_grad_mag,
        }

    mean = weighted_mean(values, weights)
    median = weighted_percentile(values, weights, 50)
    std = weighted_std(values, weights)
    pct_vals = weighted_percentile(values, weights, percentiles)

    return {
        "x_pix_lowres": x_pix_lowres,
        "y_pix_lowres": y_pix_lowres,
        "total_flux": total_flux,
        "mean": mean,
        "median": median,
        "std": std,
        "percentiles": {p: v for p, v in zip(percentiles, pct_vals)},
        "values": values,
        "weights": weights,
        "weights_map": weights_map,
        'impact_parameter_centre': impact_parameter_centre,
        'impact_parameter_centre_error': impact_parameter_centre_error,
        'impact_parameter_error_psf': impact_parameter_error_psf,
        'local_gradient_magnitude': weighted_grad_mag,
    }

def azimuthal_angle_stats_for_binned_pixel(
    highres_image,
    lowres_image,
    azimuthal_angle_map,
    highres_psf_FWHM,
    lowres_psf_FWHM,
    x_pix_lowres,
    y_pix_lowres,
    arc_mask_highres=None,
    percentiles=(5, 16, 25, 50, 75, 84, 95),
    normalize_weights=True
):
    """
    Compute the flux-contribution-weighted azimuthal angle distribution
    for one binned spaxel.

    Parameters
    ----------
    highres_image : mpdaf.obj.Image-like
        High-resolution image on the same pixel grid as azimuthal_angle_map.
    lowres_image : mpdaf.obj.Image-like
        Low-resolution image.
    azimuthal_angle_map : 2D ndarray
        Azimuthal angle map on the high-resolution grid. Must match highres_image.data shape.
    highres_psf_FWHM : float
        Gaussian FWHM of the high-resolution PSF in arcsec.
    lowres_psf_FWHM : float
        Gaussian FWHM of the low-resolution PSF in arcsec.
    x_pix_lowres, y_pix_lowres : int
        Coordinates of the selected binned spaxel in the low-resolution grid.
    arc_mask_highres : 2D bool/int array or None
        Optional arc mask on HST grid.
    percentiles : tuple
        Percentiles to compute.
    normalize_weights : bool
        If True, use normalized contribution map as weights.

    Returns
    -------
    result : dict
        Weighted stats and arrays.
    """
    contrib_map, weighted_flux_map, total_flux, sens_map = compute_input_contribution_map_for_binned_pixel(
        highres_image=highres_image,
        lowres_image=lowres_image,
        highres_psf_FWHM=highres_psf_FWHM,
        lowres_psf_FWHM=lowres_psf_FWHM,
        x_pix_lowres=x_pix_lowres,
        y_pix_lowres=y_pix_lowres,
        arc_mask_highres=arc_mask_highres,
        normalize=normalize_weights
    )

    weights_map = contrib_map if normalize_weights else weighted_flux_map

    if azimuthal_angle_map.shape != weights_map.shape:
        raise ValueError("azimuthal_angle_map and hst_cutout/contribution map must have the same shape")
    
    values = np.asarray(azimuthal_angle_map, dtype=float).ravel()
    weights = np.asarray(weights_map, dtype=float).ravel()  
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[good]
    weights = weights[good]

    x_corners = [x_pix_lowres-0.5, x_pix_lowres-0.5, x_pix_lowres+0.5, x_pix_lowres+0.5]
    y_corners = [y_pix_lowres-0.5, y_pix_lowres+0.5, y_pix_lowres-0.5, y_pix_lowres+0.5]
    ra_centre, dec_centre = lowres_image.wcs.wcs.wcs_pix2world(x_pix_lowres, y_pix_lowres, 0)
    x_centre_highres, y_centre_highres = highres_image.wcs.wcs.wcs_world2pix(ra_centre, dec_centre, 0)
    azimuthal_angle_centre = azimuthal_angle_map[y_centre_highres.astype(int), x_centre_highres.astype(int)]

    az_angle_corners = []
    for x_corner, y_corner in zip(x_corners, y_corners):
        ra_corner, dec_corner = lowres_image.wcs.wcs.wcs_pix2world(x_corner, y_corner, 0)
        x_corner_highres, y_corner_highres = highres_image.wcs.wcs.wcs_world2pix(ra_corner, dec_corner, 0)
        az_angle_corners.append(azimuthal_angle_map[y_corner_highres.astype(int), x_corner_highres.astype(int)])
    azimuthal_angle_centre_error = np.ptp(az_angle_corners)/2



    if len(values) == 0:
        return {
            "x_pix_lowres": x_pix_lowres,
            "y_pix_lowres": y_pix_lowres,
            "total_flux": total_flux,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "percentiles": {p: np.nan for p in percentiles},
            "values": values,
            "weights": weights,
            "weights_map": weights_map,
            'azimuthal_angle_centre': azimuthal_angle_centre,
            'azimuthal_angle_centre_error': azimuthal_angle_centre_error,
        }
    
    C = np.sum(weights*np.cos(np.radians(values))) / np.sum(weights)
    S = np.sum(weights*np.sin(np.radians(values))) / np.sum(weights)
    mean = np.degrees(np.arctan2(S, C)) % 360

    mean = weighted_mean(values, weights)
    median = weighted_percentile(values, weights, 50)
    std = weighted_std(values, weights)
    pct_vals = weighted_percentile(values, weights, percentiles)

    dx, dy = highres_image.wcs.wcs.proj_plane_pixel_scales()[0].to_value('arcsec'), highres_image.wcs.wcs.proj_plane_pixel_scales()[1].to_value('arcsec')
    phi = np.deg2rad(azimuthal_angle_map.data)
    u = np.cos(phi)
    v = np.sin(phi)

    du_dy, du_dx = np.gradient(u, dy, dx)
    dv_dy, dv_dx = np.gradient(v, dy, dx)
    grad_mag = np.sqrt(du_dx**2 + du_dy**2 + dv_dx**2 + dv_dy**2)

    weighted_grad_mag = np.sum(contrib_map * grad_mag) / np.sum(contrib_map)

    return {
        "x_pix_lowres": x_pix_lowres,
        "y_pix_lowres": y_pix_lowres,
        "total_flux": total_flux,
        "mean": mean,
        "median": median,
        "std": std,
        "percentiles": {p: v for p, v in zip(percentiles, pct_vals)},
        "values": values,
        "weights": weights,
        "weights_map": weights_map,
        'azimuthal_angle_centre': azimuthal_angle_centre,
        'azimuthal_angle_centre_error': azimuthal_angle_centre_error,
        'local_gradient_magnitude': weighted_grad_mag,
    }
