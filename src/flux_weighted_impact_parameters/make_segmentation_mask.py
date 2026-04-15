from mpdaf.obj import Image


import matplotlib.pyplot as plt
import numpy as np

from astropy.stats import sigma_clipped_stats
from astropy.io import fits
import astropy.units as u

from .segmentation_utils import (
    build_segmentation_map,
    keep_sources_within_radius,
    plot_segmentation_with_labels,
)

def make_segmentation_map_and_mask(image_cutout, box_size=40, filter_size=3, fwhm=2.0, threshold_sigma=2.0,
                                    npixels=20, radius_arcsec_sources=20, interactive=True, labels=None):

    segm, catalog, data_sub = build_segmentation_map(
        image_cutout.data,
        box_size=box_size,
        filter_size=filter_size,
        fwhm=fwhm,
        threshold_sigma=threshold_sigma,
        npixels=npixels
    )
    pixscale = image_cutout.wcs.wcs.proj_plane_pixel_scales()[0].to_value('arcsec')
    segmap = segm.data

    plot_segmentation_with_labels(
        image_cutout.data,
        segmap,
        catalog
    )

    if interactive:
        print("Enter segmentation labels to keep (comma separated), or 'all' to keep all:")
        user_input = input()
        if user_input.lower() != 'all':
            labels = [int(label.strip()) for label in user_input.split(',')]
            segmap = np.where(np.isin(segmap, labels), segmap, 0)
    elif labels is not None:
        # labels is a list

        segmap = np.where(np.isin(segmap, labels), segmap, 0)

    segmap = segmap.astype(int)
    mask_arc = (segmap > 0).astype(int)


    return segmap, catalog, mask_arc

def save_segmentation_mask_as_fits(segm, mask, image_cutout, output_filename):
    """
    Save the segmentation map and mask as a FITS file with appropriate WCS and metadata.
    """
    hdr = image_cutout.data_header.copy()
    hdr["BUNIT"] = "1 (mask), label (segmentation)"
    hdr["CTYPE1"] = image_cutout.data_header.get("CTYPE1", hdr.get("CTYPE1", ""))
    hdr["CTYPE2"] = image_cutout.data_header.get("CTYPE2", hdr.get("CTYPE2", ""))
    hdr["COMMENT"] = "Segmentation map and mask for gravitational arcs"
    hdr["COMMENT"] = f"Segmentation labels: {np.unique(segm[segm > 0])}"
    hdr["COMMENT"] = "Mask value 1 indicates pixels belonging to the arcs."

    # Create a 3D array to store both segmap and mask
    combined_data = np.zeros((2, segm.shape[0], segm.shape[1]), dtype=np.int16)
    combined_data[0] = segm
    combined_data[1] = mask

    fits.writeto(output_filename, combined_data, header=hdr, overwrite=True)
    print(f"Saved segmentation map and mask to: {output_filename}")



if __name__ == "__main__":
    HST_filename = '/data/estudiantes/jhernandez/SGAS1429+1202/HST/MAST_2025-09-06T02_22_19.050Z/HST/hst_skycell-p1555x15y20_wfc3_uvis_f475x_all_drc.fits'
    HST_ext = 1

    RA_CENTER, DEC_CENTER =  217.4779917, 12.0435263
    cutout_size = 30 # arcsec
    output_fits = "segmentation_mask.fits"

    hst_image = Image(HST_filename, ext=HST_ext)

    # background subtraction
    mean, median, std = sigma_clipped_stats(hst_image.data, sigma=3)
    hst_image.data -= median

    hst_cutout = hst_image.subimage(center=(DEC_CENTER, RA_CENTER), size=cutout_size,
                                    unit_center=u.deg, unit_size=u.arcsec)

    segmap, mask = make_segmentation_map_and_mask(hst_cutout)

    save_segmentation_mask_as_fits(segmap, mask, hst_cutout, output_fits)


