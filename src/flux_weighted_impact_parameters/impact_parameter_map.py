#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_impact_parameter_map.py

Description
-----------
This script computes a map of impact parameters in kpc relative to the G1 galaxy for each pixel in the HST image, using the lensing
deflection matrices to delens the pixel coordinates.
The resulting impact parameter map is saved as a FITS file with appropriate WCS and metadata.

Main tasks
----------
- Read an image (e.g., HST) and its WCS
- Delens pixel coordinates using deflection matrices
- Compute impact parameters in kpc relative to G1 in the source plane
- Save the impact parameter map as a FITS file

Usage
-----
python make_impact_parameter_map.py

Parameters
-----
The parameters are defined in the `if __name__ == "__main__":` block at the end of the script, including:
- `dir_matrices`: Directory containing the deflection matrices
- `alpha_x_filename`, `alpha_y_filename`: Filenames of the deflection matrices
- `image_filename`: Path to the image file
- `image_data_ext`: Extension of the image data
- `z_norm_matrices`: Redshift at which the deflection matrices are normalized
- `z_lens`: Redshift of the lens
- `z_source`: Redshift of the source
- `ra_G1`, `dec_G1`: RA and Dec of G1 in degrees
- `output_fits`: Filename for the output FITS file

Assumptions
-----
- The deflection matrices are in arcsec and are aligned with the WCS of the input image.
- The input image has a valid WCS that can be used to convert pixel coordinates to sky coordinates.
- The cosmology used for kpc conversion is FlatLambdaCDM with H0=70 km/s/Mpc and Om0=0.3.
- The deflection matrices are calculated at the redshift you want to compute the impact parameters for (e.g., source redshift).
- Deflections can be scaled if matrices are contructed at a different redshift than the source, but this is only valid in single-plane lenses.

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


def impact_parameter_kpc(ra1_deg, dec1_deg, ra2_deg, dec2_deg, z, cosmo=FlatLambdaCDM(H0=70, Om0=0.3)):
    """
    Proper impact parameter in kpc at redshift z.
    """
    ra1 = np.asarray(ra1_deg, dtype=float)
    dec1 = np.asarray(dec1_deg, dtype=float)
    ra2 = np.asarray(ra2_deg, dtype=float)
    dec2 = np.asarray(dec2_deg, dtype=float)

    dra_arcsec = (ra1 - ra2) * np.cos(np.deg2rad(0.5 * (dec1 + dec2))) * 3600.0
    ddec_arcsec = (dec1 - dec2) * 3600.0
    theta_arcsec = np.hypot(dra_arcsec, ddec_arcsec)

    kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(z).to(u.kpc / u.arcsec)
    return (theta_arcsec * kpc_per_arcsec.value)


def make_impact_parameter_map(
        image,
        ra_G, dec_G, z_G,
        dir_matrices='.',
        alpha_x_filename='alpha_x.fits',
        alpha_y_filename='alpha_y.fits',
        output_fits='impact_parameter_map.fits',
        image_data_ext=0,
        cosmo=FlatLambdaCDM(H0=70, Om0=0.3),):

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

    # ============================================================
    # Save as FITS
    # ============================================================
    hdr = hst_image.data_header.copy()
    hdr["BUNIT"] = "kpc"
    hdr["CTYPE1"] = hst_image.data_header.get("CTYPE1", hdr.get("CTYPE1", ""))
    hdr["CTYPE2"] = hst_image.data_header.get("CTYPE2", hdr.get("CTYPE2", ""))
    hdr["COMMENT"] = "Impact parameter map relative to G1, computed in source plane"
    hdr["COMMENT"] = f"G1 image-plane position: RA={ra_G:.7f} deg, Dec={dec_G:.7f} deg"
    hdr["COMMENT"] = f"G1 source-plane position: RA={ra_G_delensed:.7f} deg, Dec={dec_G_delensed:.7f} deg"
    hdr["COMMENT"] = f"Source redshift used for kpc conversion: z={z_G}"

    fits.writeto(output_fits, b_kpc_map.astype(np.float32), header=hdr, overwrite=True)

    print(f"Saved: {output_fits}")
    print(f"Map shape: {b_kpc_map.shape}")
    print(f"Impact parameter range: {np.nanmin(b_kpc_map):.3f} - {np.nanmax(b_kpc_map):.3f} kpc")
    print(f"G1 source-plane position: RA={ra_G_delensed:.7f}, Dec={dec_G_delensed:.7f}")


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
