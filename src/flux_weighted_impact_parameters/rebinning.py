from mpdaf.obj import Image
import numpy as np
import astropy.units as u
from skimage.draw import polygon



def downscale_highres_image(highres_image: Image, 
                            target_pixscale_arcsec: float) -> Image:
    wcs_input = highres_image.wcs
    ny_input, nx_input = highres_image.data.shape

    dy_arcsec, dx_arcsec = wcs_input.get_axis_increments(u.arcsec)
    if (np.abs(dy_arcsec) - np.abs(dx_arcsec)) > 1e-10:
        raise ValueError("Input image has non-square pixels, which is not supported.")

    input_pixscale_arcsec = np.abs(dx_arcsec)

    # Calculate the rebinning factor
    rebin_factor = np.round(target_pixscale_arcsec / input_pixscale_arcsec)

    rebin_factor_int = int(rebin_factor)

    # the image will be cropped to the largest size that is divisible by the rebinning factor
    new_dim = (ny_input // rebin_factor_int, nx_input // rebin_factor_int)


    # now define the newstart at the center of the new pixel grid to match the corresponding position
    # in the original image
    # this what the images will be aligned at the bottom left corner of the new pixel grid
    centre_x = (rebin_factor - 1 ) / 2
    centre_y = centre_x

    dec_start, ra_start = wcs_input.pix2sky((centre_x, centre_y))[0]
    new_start = (dec_start, ra_start)


    rebinned_image = highres_image.regrid(newdim=new_dim, refpos=new_start, refpix=[0, 0], newinc=target_pixscale_arcsec, flux=True, unit_inc=u.arcsec)
    return rebinned_image


def upsample_low_res_mask(low_res_mask_image: Image, high_res_image: Image) -> Image:
    wcs_low  = low_res_mask_image.wcs
    wcs_high = high_res_image.wcs
    ny_high, nx_high = high_res_image.data.shape

    # All masked pixel indices
    ys_low, xs_low = np.where(low_res_mask_image.data > 0)
    if len(ys_low) == 0:
        return Image(np.zeros((ny_high, nx_high), dtype=np.uint8), wcs=wcs_high)

    # Build all 4 corners for every masked pixel: shape (N, 4)
    # corners in low-res pixel space: (x, y) pairs
    corners_x = np.stack([xs_low - 0.5,     xs_low + 0.5, xs_low + 0.5, xs_low - 0.5    ], axis=1)  # (N,4)
    corners_y = np.stack([ys_low - 0.5,     ys_low - 0.5,     ys_low + 0.5, ys_low + 0.5], axis=1)  # (N,4)

    # Flatten → project all corners at once → reshape back
    flat_x = corners_x.ravel()
    flat_y = corners_y.ravel()

    dec_corners, ra_corners = wcs_low.pix2sky(np.array([flat_y, flat_x]).T).T
    hy_corners, hx_corners = wcs_high.sky2pix(np.array([dec_corners, ra_corners]).T).T

    hx_corners = hx_corners.reshape(-1, 4)  # (N, 4)
    hy_corners = hy_corners.reshape(-1, 4)  # (N, 4)

    # Fill polygons on the output mask
    mask_data = np.zeros((ny_high, nx_high), dtype=np.uint8)

    for i in range(len(ys_low)):
        px = hy_corners[i]   # row coords (y) for skimage
        py = hx_corners[i]   # col coords (x) for skimage
        rr, cc = polygon(px, py, shape=(ny_high, nx_high))
        mask_data[rr, cc] = 1

    return Image(data=mask_data, wcs=wcs_high)

def get_extent(image: Image) -> list:
    """Return (left, right, bottom, top) in pixel units of the image's own grid."""
    ny, nx = image.data.shape
    return [-0.5, nx - 0.5, -0.5, ny - 0.5]

def get_extent_in_other_frame(source_image: Image, target_image: Image) -> list:
    """
    Return the extent of source_image expressed in target_image pixel coordinates.
    Projects the 4 corners of source_image through WCS into target_image pixel space.
    """
    ny_src, nx_src = source_image.data.shape
    # Four corners in source pixel space
    corners_src = np.array([[-0.5, -0.5], [nx_src - 0.5, -0.5], [nx_src - 0.5, ny_src - 0.5], [-0.5, ny_src - 0.5]], dtype=float)
    dec, ra = source_image.wcs.pix2sky(np.array([corners_src[:, 0], corners_src[:, 1]]).T).T
    x_tgt, y_tgt = target_image.wcs.sky2pix(np.array([dec, ra]).T).T
    return [x_tgt.min(), x_tgt.max(), y_tgt.min(), y_tgt.max()]


def make_pixel_geometric_mask(highres_image: Image, 
                              lowres_image: Image, 
                              x_pix_lowres: int, y_pix_lowres: int) -> np.ndarray:
    """
    Create a geometric mask for a given pixel in the low-resolution image.
    The mask is 1 for high-res pixels that fall within the area of the low-res pixel, and 0 otherwise.
    """
    mask_data = np.zeros_like(highres_image.data, dtype=np.uint8)
    mask_lowres_data = np.zeros_like(lowres_image.data, dtype=np.uint8)
    mask_lowres_data[y_pix_lowres, x_pix_lowres] = 1
    mask_lowres_image = Image(data=mask_lowres_data, wcs=lowres_image.wcs)
    upsampled_mask = upsample_low_res_mask(mask_lowres_image, highres_image)
    return upsampled_mask



if __name__ == "__main__":
    from astropy.stats import sigma_clipped_stats
    import matplotlib.pyplot as plt

    test_image = "/data/estudiantes/jhernandez/SGAS1429+1202/NOTEBOOKS/tomography/flux_weighted_impact_parameters/test_on_J1226/f606w.fits"
    RA_CENTER, DEC_CENTER = 186.7137862, 21.8715231
    cutout_size =20 # arcsec
    MUSE_binning_factor = 4
    MUSE_native_pixscale = 0.2 # arcsec/pix

    PSF_MUSE = 0.8 # arcsec
    PSF_HST = 0.112 # arcsec

    # ============== Open high resolution image ==============
    print("Opening high resolution image...")
    # image filename
    hst_image = Image(test_image, ext=1)
    mean, median, std = sigma_clipped_stats(hst_image.data, sigma=3)
    hst_image.data -= median
    hst_pixscale = hst_image.wcs.wcs.proj_plane_pixel_scales()[0].to_value(u.arcsec)

    hst_cutout = hst_image.subimage(center=(DEC_CENTER, RA_CENTER), size=cutout_size,
                                        unit_center=u.deg, unit_size=u.arcsec)
    
    # downscale to a 0.8 arc pixel grid
    rebinned_image = downscale_highres_image(hst_cutout, target_pixscale_arcsec=MUSE_binning_factor * MUSE_native_pixscale)

    #mask an example pixel
    mask_image = np.zeros_like(rebinned_image.data, dtype=np.uint8)
    mask_image[5, 5] = 1
    mask_image = Image(data=mask_image, wcs=rebinned_image.wcs)
    upsampled_mask = upsample_low_res_mask(mask_image, hst_cutout)

    extent_hst = get_extent(hst_cutout)
    extent_rebinned = get_extent(rebinned_image)
    extent_hst_in_rebinned = get_extent_in_other_frame(hst_cutout, rebinned_image)
    extent_rebinned_in_hst = get_extent_in_other_frame(rebinned_image, hst_cutout)

    # make sanity plots
    # firs plot the original image and the rebinned image with contours of the original image plot images with origin = lower
    # align the contours assuming images are aligned at their bottom left corner pixel

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(hst_cutout.data, origin='lower', cmap='gray', extent=extent_hst, vmin=-1*std, vmax=5*std)
    ax[0].set_title("Original HST Cutout")
    ax[1].imshow(rebinned_image.data, origin='lower', cmap='gray', extent=extent_rebinned_in_hst)
    ax[1].contour(hst_cutout.data, colors='red', origin='lower', extent=extent_hst, levels=[5*std, 10*std, 30*std])
    ax[1].set_title("Rebinned Image (0.8 arcsec/pix)")
    plt.tight_layout()

    # now plot the upsampled mask, overplotting the original mask pixel
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(upsampled_mask.data, origin='lower', cmap='gray', extent=extent_hst)
    ax.imshow(mask_image.data, origin='lower', cmap='Reds', extent=extent_rebinned_in_hst, alpha=0.5)
    ax.set_title("Upsampled Mask (white) over Original Mask (red)")
    plt.show()




                    




