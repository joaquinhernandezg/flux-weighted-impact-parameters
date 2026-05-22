from typing import Union
from .lensing import make_delens
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from mpdaf.obj import Image
import astropy.units as u
import numpy as np


def calculate_azimuthal_angle(ra: np.ndarray, 
                              dec: np.ndarray, 
                              ra_G: float, 
                              dec_G: float, 
                              PA_G: float)-> np.ndarray:
    """
    Calculate the azimuthal angle in degrees between the position of G1 (ra_G, dec_G) and the positions of the pixel centers (ra, dec),
    with respect to the major axis of G1 defined by its position angle PA_G (in degrees, measured east of north).
    The azimuthal angle is defined such that 0 degrees corresponds to the major axis of G1, and it increases counterclockwise (i.e., east of north).

    Parameters
    ----------
    ra : np.ndarray
        Right Ascension of the pixel centers in degrees.
    dec : np.ndarray
        Declination of the pixel centers in degrees.
    ra_G : float
        Right Ascension of G1 in degrees.
    dec_G : float
        Declination of G1 in degrees.
    PA_G : float
        Position angle of G1 in degrees, measured east of north.
    
    Returns
    -------
    phi_deg : np.ndarray
        Azimuthal angle in degrees for each pixel center, where 0 degrees is along the major axis of G1 and increases counterclockwise.

    """

    dra = (ra - ra_G) * np.cos(np.deg2rad(dec_G)) * 3600.0
    ddec = (dec - dec_G) * 3600.0
    theta = np.arctan2(ddec, -dra)  # angle in radians east of north
    PA_rad = np.radians(PA_G)
    phi = (theta - PA_rad) % (2 * np.pi)  # azimuthal angle in radians, 0 along major axis, increasing counterclockwise, east of north
    return np.degrees(phi)



def make_azimuthal_angle_map(
        image: Union[str, Image],
        ra_G: float,
        dec_G: float, 
        z_G: float,
        PA_G: float,
        dir_matrices: str='.',
        alpha_x_filename: str='alpha_x.fits',
        alpha_y_filename: str='alpha_y.fits',
        output_fits: str='azimuthal_angle_map.fits',
        image_data_ext=0,
        cosmo=FlatLambdaCDM(H0=70, Om0=0.3),):

    # set header key BUNIT to 'arcsec'
    fits.setval(f"{dir_matrices}/{alpha_x_filename}", "BUNIT", value="arcsec")
    fits.setval(f"{dir_matrices}/{alpha_y_filename}", "BUNIT", value="arcsec")
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

    # Delens all HST pixel centers
    ra_delensed, dec_delensed = make_delens(ra_centers, dec_centers, dir_matrices=dir_matrices,
                                           alpha_x_filename=alpha_x_filename, alpha_y_filename=alpha_y_filename)

    # Delens G1 position
    ra_G_delensed, dec_G_delensed = make_delens(ra_G, dec_G, dir_matrices=dir_matrices,
                                               alpha_x_filename=alpha_x_filename, alpha_y_filename=alpha_y_filename)
    ra_G_delensed = ra_G_delensed[0]
    dec_G_delensed = dec_G_delensed[0]

    # compute azimuthal angle map in degrees, where 0 deg is along the major axis of G1 and increases counterclockwise
    phi_deg = calculate_azimuthal_angle(ra_delensed, dec_delensed, ra_G_delensed, dec_G_delensed, PA_G)
    phi_map = phi_deg.reshape(ny, nx)

    phi_image = hst_image.copy()
    phi_image.data = phi_map

    if output_fits is not None:
        phi_image.write(output_fits)

    return phi_image


