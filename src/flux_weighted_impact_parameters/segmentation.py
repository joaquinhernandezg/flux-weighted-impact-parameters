
import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_sources, deblend_sources, SourceCatalog
from photutils.segmentation import SegmentationImage
from astropy.stats import sigma_clipped_stats
from astropy.io import fits
import astropy.units as u
from mpdaf.obj import Image
from collections import defaultdict
from astropy.table import Table



def build_segmentation_map(
        image: np.ndarray,
        box_size: int = 40,
        filter_size: float = 3.0,
        fwhm: float = 2.0,
        threshold_sigma: float = 2.0,
        npixels: int = 20,
        deblend: bool = True,
        nlevels: int = 2,
        contrast: float = 0.01):
    """
    Build a segmentation map from an astronomical image.

    Parameters
    ----------
    image : 2D numpy array
        Input image (e.g., HST cutout).
    box_size : int
        Background estimation box size.
    filter_size : int
        Background smoothing filter.
    fwhm : float
        FWHM of Gaussian kernel in pixels used for detection smoothing.
    threshold_sigma : float
        Detection threshold in units of background RMS.
    npixels : int
        Minimum number of connected pixels for detection.
    deblend : bool
        Whether to run source deblending.
    nlevels : int
        Number of multi-thresholding levels for deblending.
    contrast : float
        Minimum contrast ratio for deblending.

    Given that this is intended for segmentation on extended gravitational arcs
    a low nlevels and contrast are recommended to avoid over-deblending. 
    Large npixels are also recommended to capture large arc structures.

    Returns
    -------
    segm : SegmentationImage
        Photutils segmentation map.
    catalog : SourceCatalog
        Photutils catalog of detected sources.
    data_sub : 2D array
        Background-subtracted image used for detection.
    """

    data = image.astype(float)

    # Replace non finite pixels
    data[~np.isfinite(data)] = np.nanmedian(data)

    # --------------------------------------------------
    # Background estimation and subtraction
    # --------------------------------------------------
    bkg = Background2D(
        data,
        box_size=(box_size, box_size),
        filter_size=(filter_size, filter_size),
        bkg_estimator=MedianBackground(),
        exclude_percentile=10.0
    )

    data_sub = data - bkg.background

    # --------------------------------------------------
    # Convolution for detection
    # --------------------------------------------------
    sigma_pix = fwhm / 2.355
    kernel = Gaussian2DKernel(sigma_pix)
    kernel.normalize()

    data_conv = convolve(data_sub, kernel)

    # --------------------------------------------------
    # Detection threshold
    # --------------------------------------------------
    threshold = threshold_sigma * bkg.background_rms

    segm = detect_sources(data_conv, threshold, npixels=npixels)

    if segm is None:
        raise RuntimeError("No sources detected. Try lowering threshold.")

    # --------------------------------------------------
    # Deblend
    # --------------------------------------------------
    if deblend:
        segm = deblend_sources(
            data_conv,
            segm,
            npixels=npixels,
            nlevels=nlevels,
            contrast=contrast,
            progress_bar=False
        )

    # --------------------------------------------------
    # Build catalog
    # --------------------------------------------------
    catalog = SourceCatalog(data_sub, segm)

    return segm, catalog, data_sub


def keep_sources_within_radius(segm: SegmentationImage, 
                               catalog: SourceCatalog, 
                               radius_pix: float = 100):
    """
    Remove segmentation labels whose centroids are farther than
    `radius_pix` pixels from the image center. This is
    useful to keep only sources near the lensing galaxy and arcs.

    Parameters
    ----------
    segm : photutils.segmentation.SegmentationImage
        Segmentation map from photutils.
    catalog : photutils.segmentation.SourceCatalog
        Catalog associated with the segmentation map.
    radius_pix : float
        Radius in pixels around the center to keep sources.

    Returns
    -------
    segm_clean : SegmentationImage
        Segmentation map with distant sources removed.
    """

    segmap = segm.copy()

    ny, nx = segmap.shape
    x0 = (nx - 1) / 2
    y0 = (ny - 1) / 2

    tbl = catalog.to_table()

    labels_to_keep = []

    for row in tbl:
        x = row['xcentroid']
        y = row['ycentroid']

        r = np.sqrt((x - x0)**2 + (y - y0)**2)

        if r <= radius_pix:
            labels_to_keep.append(int(row['label']))

    segmap_clean = np.zeros_like(segmap)

    for label in labels_to_keep:
        segmap_clean[segmap == label] = label

    segm_clean = segm.copy()
    segm_clean.data = segmap_clean

    return segm_clean


def plot_segmentation_with_labels(image: np.ndarray, 
                                  segm: SegmentationImage, 
                                  catalog: SourceCatalog,
                                  percentile: tuple = (5, 99.5),
                                  fontsize: int = 10,
                                  cmap: str = 'gray'):
    """
    Plot image + segmentation map with labels visible against background.
    Useful for visualizing the segmentation results and select the segmentation labels.

    Parameters
    ----------
    image : 2D array
        Image used for segmentation (e.g., HST cutout).
    segm : SegmentationImage
        Photutils segmentation object.
    catalog : SourceCatalog
        Catalog associated with segmentation.
    percentile : tuple
        Percentile stretch for image display.
    fontsize : int
        Font size of labels.
    cmap : str
        Colormap for image display.
    """

    segmap = segm.data
    ny, nx = image.shape

    # Determine vmin and vmax for display based on percentiles of the image data
    vmin, vmax = np.percentile(image, percentile)


    fig, ax = plt.subplots(figsize=(6,6))

    ax.imshow(image, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)

    # draw contours of the segmentation map excluding the background (label=0)
    ax.contour(segmap, levels=np.unique(segmap)[1:], colors='cyan', linewidths=1)

    tbl = catalog.to_table()

    # for each segmentation label determine the centroid and plot the corresponding text
    for row in tbl:
        x = row['xcentroid']
        y = row['ycentroid']
        label = int(row['label'])

        xi = int(round(x))
        yi = int(round(y))

        # estimate local brightness
        if 0 <= xi < nx and 0 <= yi < ny:
            val = image[yi, xi]
        else:
            val = np.nanmedian(image)

        # choose opposite color for contrast
        if val > np.nanmedian(image):
            color = 'black'
        else:
            color = 'white'

        ax.text(x, y,
                f'{label}',
                color=color,
                fontsize=fontsize,
                ha='center',
                va='center',
                weight='bold')

    ax.set_xlabel('x [pix]')
    ax.set_ylabel('y [pix]')
    ax.set_title("Segmentation map with labels")

    fig.tight_layout()
    return fig, ax




def make_segmentation_map_and_mask( image_cutout: Image, 
                                    box_size: int = 40, 
                                    filter_size: int = 3, 
                                    fwhm: float = 2.0, 
                                    threshold_sigma: float = 2.0,
                                    npixels: int = 20, 
                                    radius_arcsec_sources: int = 20, 
                                    plot_segmentation: bool = True,
                                    interactive: bool = True, 
                                    labels=None):
    
    """
    Build a segmentation map and mask for the arcs in the image cutout. 
    The function first builds a segmentation map using photutils, 
    then keeps only sources within a given radius from the center,
    and finally creates a binary mask where pixels belonging to the arcs are 1 and the rest are 0.

    It is intended to be used on cutouts centered on the lensing galaxy, 
    so that the arcs are near the center and other sources (e.g., nearby galaxies)
    can be removed by keeping only sources within a certain radius. 

    A first pass of the segmentation should be done with labels=None and interactive=True 
    to visualize the segmentation and select the labels corresponding to the arcs.
    Then a second pass can be done with interactive=False and labels set to the list of labels corresponding to the arcs.

    Parameters
    ----------
    image_cutout : Image
        An MPDAF Image cutout centered on the lensing galaxy and arcs.
    box_size : int
        Box size for background estimation in photutils.
    filter_size : int
        Filter size for background estimation in photutils.
    fwhm : float
        FWHM of Gaussian kernel in pixels for detection smoothing in photutils.
    threshold_sigma : float
        Detection threshold in units of background RMS for photutils.
    npixels : int
        Minimum number of connected pixels for detection in photutils.
    radius_arcsec_sources : int
        Radius in arcseconds around the center to keep sources in the segmentation map.
    plot_segmentation : bool
        Whether to plot the segmentation map with labels for visualization.
    interactive : bool
        Whether to interactively select segmentation labels to keep. If False, use the `labels` parameter.
    labels : list or None
        List of segmentation labels to keep. If None and interactive=False, all labels will be kept.

    Returns
    -------
    segmap : 2D array
        Segmentation map with only the selected sources (arcs) labeled.
    catalog : SourceCatalog
        Catalog of detected sources from photutils.
    mask_arc : 2D array
        Binary mask where pixels belonging to the arcs are 1 and the rest are 0.    
    """

    # Build segmentation map
    segm, catalog, data_sub = build_segmentation_map(
        image_cutout.data,
        box_size=box_size,
        filter_size=filter_size,
        fwhm=fwhm,
        threshold_sigma=threshold_sigma,
        npixels=npixels
    )

    segmap = segm.data

    if plot_segmentation:
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

    Parameters
    ----------
    segm : 2D array
        Segmentation map with labeled sources.
    mask : 2D array
        Binary mask where pixels belonging to the arcs are 1 and the rest are 0.
    image_cutout : Image
        The original MPDAF Image cutout used for segmentation, to copy WCS and metadata from.
    output_filename : str
        Path to save the output FITS file containing the segmentation map and mask.

    Returns
    -------
    None
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
    HST_filename = 'data/f606w.fits'
    HST_ext = 1

    RA_CENTER, DEC_CENTER = 186.7137862, 21.8715231
    cutout_size = 20 # arcsec
    output_fits = "data/segmentation_mask.fits"

    hst_image = Image(HST_filename, ext=HST_ext)

    # background subtraction
    mean, median, std = sigma_clipped_stats(hst_image.data, sigma=3)
    hst_image.data -= median

    hst_cutout = hst_image.subimage(center=(DEC_CENTER, RA_CENTER), size=cutout_size,
                                    unit_center=u.deg, unit_size=u.arcsec)

    segmap, mask = make_segmentation_map_and_mask(hst_cutout)

    save_segmentation_mask_as_fits(segmap, mask, hst_cutout, output_fits)