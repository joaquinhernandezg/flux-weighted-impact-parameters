import numpy as np
import astropy.units as u
from astropy.stats import sigma_clipped_stats
import astropy.units as u
import matplotlib.pyplot as plt
from mpdaf.obj import Image
from .rebinning import downscale_highres_image, make_pixel_geometric_mask


def compute_arc_sky_flux_contributions(highres_cutout: Image, 
                                       arc_mask: np.ndarray, 
                                       target_pixscale: float, 
                                       psf_kernel: float):
    """
    Split the high-resolution image into arc and non-arc components, propagate both through
    PSF convolution + WCS resampling, and compute their fractional contribution
    to each binned pixel.

    The final products are 2D maps, containing the arc flux contribution, non-arc flux contribution, total flux, and relative contributions.
    After PSF convolution and resampling.

    The idea of this function is to know the relative contribution arc vs non-arc flux on a given pixel in low-spatial-resolution data,
    using as prior the high-spatial-resolution image and a segmentation mask of the arc.

    Parameters
    ----------
    highres_cutout : mpdaf.obj.Image
        Original high-resolution cutout.
    arc_mask : np.ndarray
        Binary mask of the arc in high-resolution pixel space. True/1 = arc.
    target_pixscale : float
        Target pixel scale in arcsec for the low-resolution image.
    psf_kernel : float
        Gaussian FWHM kernel in arcsec to convolve the high-resolution image to the KCWI PSF.

    Returns
    -------
    results : dict\
        Each entry is a 2D array on the low-resolution grid:
        Dictionary containing:
        - 'arc_flux'          : image of the PSF-convolved and resampled arc flux.
        - 'sky_flux'          : non-arc flux PSF-convolved and resampled.
        - 'total_flux'        : total flux PSF-convolved and resampled (should be sum of arc_flux and sky_flux).
        - 'arc_fraction_flux' : relative arc flux contribution per spaxel (arc_flux / total_flux).
        - 'sky_fraction_flux' : relative non-arc flux contribution per spaxel (sky_flux / total_flux).
        - 'arc_mask_input'    : input binary mask
    """

    data = np.array(highres_cutout.data, dtype=float)
    mask = np.array(arc_mask, dtype=bool)

    # clean non-finite values
    data = np.where(np.isfinite(data), data, 0.0)

    # split image into arc and non-arc components
    data_arc = data * mask
    data_sky = data * (~mask)
    data_total = data.copy()

    # make image copies
    img_arc = highres_cutout.copy()
    img_sky = highres_cutout.copy()
    img_total = highres_cutout.copy()

    img_arc.data = data_arc
    img_sky.data = data_sky
    img_total.data = data_total

    # convolve all components identically
    img_arc_conv = img_arc.fftconvolve_gauss(
        fwhm=(psf_kernel, psf_kernel), unit_fwhm=u.arcsec
    )
    img_sky_conv = img_sky.fftconvolve_gauss(
        fwhm=(psf_kernel, psf_kernel), unit_fwhm=u.arcsec
    )
    img_total_conv = img_total.fftconvolve_gauss(
        fwhm=(psf_kernel, psf_kernel), unit_fwhm=u.arcsec
    )
    
    # resample to KCWI grid
    arc_resampled = downscale_highres_image(img_arc_conv, target_pixscale_arcsec=target_pixscale)
    sky_resampled = downscale_highres_image(img_sky_conv, target_pixscale_arcsec=target_pixscale)
    total_resampled = downscale_highres_image(img_total_conv, target_pixscale_arcsec=target_pixscale)

    #arc_resampled = img_arc_conv.align_with_image(lowres_image)
    #sky_resampled = img_sky_conv.align_with_image(lowres_image)
    #total_resampled = img_total_conv.align_with_image(lowres_image)

    arc_flux = np.array(arc_resampled.data, dtype=float)
    sky_flux = np.array(sky_resampled.data, dtype=float)
    total_flux = np.array(total_resampled.data, dtype=float)

    mean, std = total_resampled.background(niter=3, sigma=3)
    snr = total_flux / std if std > 0 else np.zeros_like(total_flux)
    

    # avoid numerical issues
    eps = np.finfo(float).eps
    denom = np.where(np.abs(total_flux) > eps, total_flux, np.nan)

    arc_fraction_flux = arc_flux / denom
    sky_fraction_flux = sky_flux / denom

    # numerical cleanup
    arc_fraction_flux = np.clip(arc_fraction_flux, 0, 1)
    sky_fraction_flux = np.clip(sky_fraction_flux, 0, 1)

    return {
        "arc_flux": arc_flux,
        "sky_flux": sky_flux,
        "total_flux": total_flux,
        "arc_flux_fraction": arc_fraction_flux,
        "sky_flux_fraction": sky_fraction_flux,
        "arc_mask_input": mask,
        "snr": snr,
    }



def compute_input_contribution_map_for_binned_pixel(
    highres_image: Image,
    lowres_image: Image,
    highres_psf_FWHM: float,
    lowres_psf_FWHM: float,
    x_pix_lowres: int,
    y_pix_lowres: int,
    arc_mask_highres=None):

    """
    Compute the contribution map on the original high-resolution grid for a single
    binned/output pixel.

    Parameters
    ----------
    highres_image : mpdaf.obj.Image-like
        Original high-resolution image.
    lowres_image : mpdaf.obj.Image-like
        Output/binned image defining the target grid (e.g. a MUSE image).
    highres_psf_FWHM : float
        Gaussian FWHM in arcsec used to degrade high-resolution to target PSF.
    lowres_psf_FWHM : float
        Gaussian FWHM in arcsec used for the target PSF.
    x_pix_lowres, y_pix_lowres : int
        Target pixel coordinates on the lowresolution image.
    arc_mask_highres : 2D bool/int array or None
        If given, restrict contributions to arc-mask pixels only.

    Returns
    -------
    flux_contribution_weights : 2D array
        The contribution map on the original high-resolution grid for the selected binned pixel, normalized to sum to 1.
    flux_contribution_map : 2D array
        The contribution map on the original high-resolution grid for the selected binned pixel, in flux units (i.e. multiplied by the highres image).
    total_flux : float
        The total flux contribution to the selected binned pixel (sum of flux_contribution_map).
    sens_map : 2D array
        The sensitivity map on the high-resolution grid for the selected binned pixel, i.e. the PSF-convolved and resampled footprint of the selected binned pixel on the high-resolution grid.
    footprint_img : 2D array
        The geometric footprint of the selected binned pixel on the high-resolution grid, before PSF convolution.
    """

    # --------------------------------------------------
    # 1. Calculate the effective PSF kernel to apply to the high-resolution image to match the low-resolution PSF.
    # --------------------------------------------------
    psf_kernel_arcsec = np.sqrt(max(0, lowres_psf_FWHM**2 - highres_psf_FWHM**2))

    # --------------------------------------------------
    # 1. Delta image on target grid
    # --------------------------------------------------

    # --------------------------------------------------
    # 2. Resample highres image to target grid (with PSF convolution)
    # This creates a map of which highres pixels contribute to the selected lowres
    # pixel, and with which weight (the "footprint").
    # Is a mask that is 1 where the selected lowres pixel receives flux from the
    # highres image, and 0 elsewhere. Assumming a perfect geometric mapping
    # and no PSF convolution, this would be a binary mask of the highres pixels
    # that fall within the lowres pixel.
    # --------------------------------------------------
    footprint_img = make_pixel_geometric_mask(highres_image, lowres_image, x_pix_lowres, y_pix_lowres)

    # --------------------------------------------------
    # 3. Convolve footprint with same PSF
    #    This gives the HST-grid sensitivity map for this output
    # By convolving the footprint with the same PSF kernel,
    # we account for the fact that the flux from a given highres pixel can
    # spread to neighboring lowres pixels due to the PSF.
    # results in a sensitivity map on the highres grid that indicates how much
    # highres pixel contributes to the selected lowres pixel after PSF convolution.
    # --------------------------------------------------

    sensitivity = footprint_img.fftconvolve_gauss(
        fwhm=(psf_kernel_arcsec, psf_kernel_arcsec),
        unit_fwhm=u.arcsec
    )

    sens_map = np.array(sensitivity.data, dtype=float)
    sens_map = np.where(np.isfinite(sens_map), sens_map, 0.0)

    # --------------------------------------------------
    # 4. Multiply by original HST image
    # By multiplying the sensitivity map by the original high-resolution image,
    # we get the flux contribution map, which indicates how much flux from
    # each highres pixel contributes to the selected lowres pixel after
    # PSF convolution and resampling.
    # When providing an arc mask, we restrict the contributions to the arc pixels only.
    # --------------------------------------------------
    highres_data = np.array(highres_image.data, dtype=float)
    highres_data = np.where(np.isfinite(highres_data), highres_data, 0.0)

    if arc_mask_highres is not None:
        mask = np.array(arc_mask_highres, dtype=bool)
        highres_data_use = highres_data * mask
    else:
        highres_data_use = highres_data

    flux_contribution_map = highres_data_use * sens_map
    total_flux = np.nansum(flux_contribution_map)

    flux_contribution_weights  = flux_contribution_map.copy()
    flux_contribution_weights  /= np.nansum(flux_contribution_weights ) 

    return flux_contribution_weights , flux_contribution_map, total_flux, sens_map, footprint_img

