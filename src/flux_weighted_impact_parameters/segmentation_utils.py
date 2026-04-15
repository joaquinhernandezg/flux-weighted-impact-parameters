
import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_sources, deblend_sources, SourceCatalog


def build_segmentation_map(
        image,
        box_size=40,
        filter_size=3,
        fwhm=2.0,
        threshold_sigma=2.0,
        npixels=20,
        deblend=True,
        nlevels=2,
        contrast=0.01):
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


def keep_sources_within_radius(segm, catalog, radius_pix=100):
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


def plot_segmentation_with_labels(image, segm, catalog,
                                  percentile=(5, 99.5),
                                  fontsize=10,
                                  cmap='gray'):
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

    vmin, vmax = np.percentile(image, percentile)

    fig, ax = plt.subplots(figsize=(6,6))

    ax.imshow(image, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)

    # segmentation contours
    ax.contour(segmap, levels=np.unique(segmap)[1:], colors='cyan', linewidths=1)

    tbl = catalog.to_table()

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

    plt.tight_layout()
    return fig, ax
