from .lensing import make_delens
from astropy.io import fits
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from mpdaf.obj import Image, Cube
import astropy.units as u
import numpy as np
import os

def calculate_azimuthal_angle(ra, dec, ra_G, dec_G, PA_G):
    dra = (ra - ra_G) * np.cos(np.deg2rad(dec_G)) * 3600.0
    ddec = (dec - dec_G) * 3600.0
    theta = np.arctan2(ddec, -dra)  # angle in radians east of north
    PA_rad = np.radians(PA_G)
    phi = (theta - PA_rad) % (2 * np.pi)  # azimuthal angle in radians, 0 along major axis, increasing counterclockwise, east of north
    return np.degrees(phi)

def make_azimuthal_angle_map(
        image,
        ra_G, dec_G, z_G, PA_G,
        dir_matrices='.',
        alpha_x_filename='alpha_x.fits',
        alpha_y_filename='alpha_y.fits',
        output_fits='azimuthal_angle_map.fits',
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

    # compute azimuthal angle map in degrees, where 0 deg is along the major axis of G1 and increases counterclockwise
    phi_deg = calculate_azimuthal_angle(ra_delensed, dec_delensed, ra_G_delensed, dec_G_delensed, PA_G)
    phi_map = phi_deg.reshape(ny, nx)

    hdr = hst_image.data_header.copy()
    hdr["BUNIT"] = "deg"
    hdr["CTYPE1"] = hst_image.data_header.get("CTYPE1", hdr.get ("CTYPE1", ""))
    hdr["CTYPE2"] = hst_image.data_header.get("CTYPE2", hdr.get("CTYPE2", ""))
    hdr["COMMENT"] = "Azimuthal angle map relative to G1 major axis, computed in source plane"
    hdr["COMMENT"] = f"G1 image-plane position: RA={ra_G:.7f} deg, Dec={dec_G:.7f} deg"
    hdr["COMMENT"] = f"G1 source-plane position: RA={ra_G_delensed:.7f} deg, Dec={dec_G_delensed:.7f} deg"
    hdr["COMMENT"] = f"G1 position angle (E of N): PA={PA_G:.1f} deg"
    hdr["COMMENT"] = f"Source redshift used for kpc conversion: z={z_G}"

    fits.writeto(output_fits, phi_map.astype(np.float32), header=hdr, overwrite=True)
    return phi_map


