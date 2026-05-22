"""
Author
------
Joaquin Hernandez-Guajardo

Created
-------
2026-03-15
"""

from astropy.io import fits
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from mpdaf.obj import Image, Cube
import astropy.units as u
import numpy as np
import os
from .lensing import make_delens
from typing import Union



def impact_parameter_kpc(ra1_deg: np.ndarray, 
                         dec1_deg: np.ndarray, 
                         ra2_deg: Union[float, np.ndarray], 
                         dec2_deg: Union[float, np.ndarray], 
                         z: float, 
                         cosmo=FlatLambdaCDM(H0=70, Om0=0.3)) -> np.ndarray:
    """
    Compute the transverse impact parameter in kpc between a set of coordinates (ra1, dec1) and a reference point (ra2, dec2) at a given redshift z.
    The reference point can be a single point (e.g., the position of G1) or an array of points 
    (e.g., the delensed positions of the pixel centers), in which case the function will compute the impact parameter between each pair of coordinates.

    Parameters
    ----------
    ra1_deg : array-like
        Right Ascension of the first set of coordinates in degrees.
    dec1_deg : array-like
        Declination of the first set of coordinates in degrees.
    ra2_deg : float or np.ndarray
        Right Ascension of the reference point in degrees.
    dec2_deg : float or np.ndarray
        Declination of the reference point in degrees.
    z : float
        Redshift at which to compute the impact parameter (used for kpc conversion).
    cosmo : astropy.cosmology instance
        Cosmology to use for kpc conversion. Default is FlatLambdaCDM with H0=70 km/s/Mpc and Om0=0.3.
    """

    ra1 = np.asarray(ra1_deg, dtype=float)
    dec1 = np.asarray(dec1_deg, dtype=float)

    ra2 = np.asarray(ra2_deg, dtype=float)
    dec2 = np.asarray(dec2_deg, dtype=float)

    if ra2.ndim == 0:
        ra2 = np.full_like(ra1, ra2)
    if dec2.ndim == 0:
        dec2 = np.full_like(dec1, dec2)
    if ra1.shape != dec1.shape or ra2.shape != dec2.shape:
        raise ValueError("Input coordinate arrays must have the same shape")

    dra_arcsec = (ra1 - ra2) * np.cos(np.deg2rad(0.5 * (dec1 + dec2))) * 3600.0
    ddec_arcsec = (dec1 - dec2) * 3600.0

    theta_arcsec = np.hypot(dra_arcsec, ddec_arcsec)

    kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(z).to(u.kpc / u.arcsec)

    return (theta_arcsec * kpc_per_arcsec.value)


def make_impact_parameter_map(
        image: Union[str, Image],
        ra_G: float, dec_G: float, z_G: float,
        dir_matrices: str='.',
        alpha_x_filename: str='alpha_x.fits',
        alpha_y_filename: str='alpha_y.fits',
        output_fits: str='impact_parameter_map.fits',
        image_data_ext: int=0,
        cosmo=FlatLambdaCDM(H0=70, Om0=0.3)) -> Image:

    # set header key BUNIT to 'arcsec'
    fits.setval(f"{dir_matrices}/{alpha_x_filename}", "BUNIT", value="arcsec")
    fits.setval(f"{dir_matrices}/{alpha_y_filename}", "BUNIT", value="arcsec")


    # ============================================================
    # Load HST image
    # ============================================================
    if isinstance(image, str):
        hst_image = Image(image, ext=image_data_ext)
    elif isinstance(image, Image):
        hst_image = image
    ny, nx = hst_image.data.shape

    # Pixel centers in HST image coordinates
    y_indices, x_indices = np.mgrid[0:ny, 0:nx]
    centers_pixel = np.column_stack([x_indices.ravel(), y_indices.ravel()])

    # MPDAF/astropy WCS from HST image
    wcs_hst = hst_image.wcs.wcs
    centers_world = wcs_hst.wcs_pix2world(centers_pixel, 0)

    # Extract RA and Dec of pixel centers
    ra_centers = centers_world[:, 0]
    dec_centers = centers_world[:, 1]

    # ============================================================
    # Delens all HST pixel centers
    # ============================================================
    ra_delensed, dec_delensed = make_delens(ra_centers, dec_centers, dir_matrices=dir_matrices,
                                           alpha_x_filename=alpha_x_filename, alpha_y_filename=alpha_y_filename)

    # Delens G1 position
    ra_G_delensed, dec_G_delensed = make_delens(ra_G, dec_G, dir_matrices=dir_matrices,
                                               alpha_x_filename=alpha_x_filename, alpha_y_filename=alpha_y_filename)
    ra_G_delensed = ra_G_delensed[0]
    dec_G_delensed = dec_G_delensed[0]

    # ============================================================
    # Compute impact parameter map in kpc
    # ============================================================
    b_kpc = impact_parameter_kpc(
        ra_delensed,
        dec_delensed,
        ra_G_delensed,
        dec_G_delensed,
        z_G,
        cosmo=cosmo
    )

    b_kpc_map = b_kpc.reshape(ny, nx)

    impact_parameter_map = hst_image.copy()
    impact_parameter_map.data = b_kpc_map

    if output_fits is not None:
        impact_parameter_map.write(output_fits)

    return impact_parameter_map


if __name__ == "__main__":



    # ============================================================
    # Inputs
    # ============================================================
    dir_matrices = "/data/estudiantes/jhernandez/SGAS1429+1202/LENSMODEL/model_shapelets_f110w"
    alpha_x_filename = "alpha_x_z2.18089.fits"
    alpha_y_filename = "alpha_y_z2.18089.fits"

    image_filename = "/data/estudiantes/jhernandez/SGAS1429+1202/HST/MAST_2025-09-06T02_22_19.050Z/HST/hst_skycell-p1555x15y20_wfc3_uvis_f475x_all_drc.fits"
    image_data_ext = 1

    z_norm_matrices = 2.18089
    z_lens = 0.5331
    z_source = 2.18089

    ra_G1, dec_G1 = 217.4785468, 12.0444573

    output_fits = "impact_parameter_map_hst_f475w.fits"

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    make_impact_parameter_map(
        image_filename=image_filename,
        ra_G=ra_G1,
        dec_G=dec_G1,
        z_G=z_source,
        dir_matrices=dir_matrices,
        alpha_x_filename=alpha_x_filename,
        alpha_y_filename=alpha_y_filename,
        image_data_ext=image_data_ext,
        cosmo=cosmo
    )
