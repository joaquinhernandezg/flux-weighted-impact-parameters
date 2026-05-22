from astropy.cosmology import FlatLambdaCDM
from mpdaf.obj import Image
import astropy.units as u
import numpy as np
import os
from scipy.ndimage import distance_transform_edt
from skimage.measure import find_contours
from scipy.ndimage import binary_dilation



def scale_lensing_matrix(lensing_matrix_filename: str, 
                         z_lens: float, 
                         z_source: float, 
                         z_norm_matrices: float, 
                         output_filename: str) -> Image:
    """
    Scale the lensing matrix from the normalization redshift to the desired lens and source redshifts.

    Let DLS the angular diameter distance between the lens and the source, 
    and DS the angular diameter distance to the source. 
    The lensing matrix is proportional to DLS/DS. 
    If the input lensing matrix is normalized at a redshift z_norm_matrices, we can compute the scaling factor as:
    (DLS(z_lens, z_source)/DS(z_source)) / (DLS(z_lens, z_norm_matrices)/DS(z_norm_matrices))
    This means "de-normalizing" the input lensing matrix to the physical deflection field, and then "re-normalizing" it to the desired redshift configuration.

    IMPORTANT:
    This assumes that the lensing matrices can be scaled, which is only true in the single-plane lensing approximation.
    If the input lensing matrix was computed using a multi-plane lensing code, the scaling should not be applied.
    The scaling only applied to deflections, convergence and shear maps (and not magnification maps).

    Eq. 2.5 from "Introduction to Gravitational Lensing Lecture scripts" by
    Massimo Meneghetti (https://www.ita.uni-heidelberg.de/~jmerten/misc/meneghetti_lensing.pdf)

    Parameters
    ----------
    lensing_matrix_filename : str
        Path to the input lensing matrix FITS file.
    z_lens : float
        Redshift of the lens.
    z_source : float
        Redshift of the source.
    z_norm_matrices : float
        Redshift at which the input lensing matrix is normalized.
    output_filename : str
        Path to save the scaled lensing matrix FITS file.

    Returns
    -------
    Image
        The scaled lensing matrix as an MPDAF Image object.
    """

    lensing_matrix = Image(lensing_matrix_filename)
    scaled_lensing_matrix = lensing_matrix.copy()

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    DLS1 = cosmo.angular_diameter_distance_z1z2(z_lens, z_norm_matrices).to_value(u.kpc)
    DS1 = cosmo.angular_diameter_distance(z_norm_matrices).to_value(u.kpc)
    DLS_DS1 = DLS1/DS1

    DLS2 = cosmo.angular_diameter_distance_z1z2(z_lens, z_source).to_value(u.kpc)
    DS2 = cosmo.angular_diameter_distance(z_source).to_value(u.kpc)
    DLS_DS2 = DLS2/DS2

    scaled_lensing_matrix.data = lensing_matrix.data * (1/DLS_DS1 * DLS_DS2)
    scaled_lensing_matrix.write(output_filename)

    return scaled_lensing_matrix


def compute_magnification_map_from_deflections(alpha_x: Image, 
                                               alpha_y: Image, save_filename=None) -> Image:
    """
    This function computes the magnification map from the deflection fields alpha_x and alpha_y.
    The magnification is given by mu = 1/detA, where A is the Jacobian matrix of the lens mapping, 
    which can be computed from the derivatives of the deflection fields.

    This function is based on Eq. 2.33 to 2.48 from "Introduction to Gravitational Lensing Lecture scripts" by
    Massimo Meneghetti (https://www.ita.uni-heidelberg.de/~jmerten/misc/meneghetti_lensing.pdf)

    This function is intended to be used with deflection fields computed at a given redshift when
    no magnification map is available, and not shear and convergence maps.

    Parameters
    ----------
    alpha_x, alpha_y : Image
        The deflection fields in the x and y directions, as MPDAF Image objects.
    save_filename : str or None
        If not None, the path to save the computed magnification map as a FITS file. If None, the magnification map is not saved to disk.
    Returns
    -------
    mu_image : Image
        The computed magnification map as an MPDAF Image object.
    """
    # pixel scale of the map
    dx = alpha_x.wcs.wcs.proj_plane_pixel_scales()[0].to_value('arcsec')
    dy = alpha_x.wcs.wcs.proj_plane_pixel_scales()[1].to_value('arcsec')

    # np.gradient returns derivatives in axis order:
    # axis 0 -> y direction
    # axis 1 -> x direction
    dax_dy, dax_dx = np.gradient(alpha_x.data, dy, dx)
    day_dy, day_dx = np.gradient(alpha_y.data, dy, dx)

    A11 = 1.0 - dax_dx
    A12 = -dax_dy
    A21 = -day_dx
    A22 = 1.0 - day_dy

    detA = A11 * A22 - A12 * A21
    mu = 1.0 / detA
    mu_image = alpha_x.copy()
    mu_image.data = mu.data
    if save_filename is not None:
        mu_image.write(save_filename)
    return mu_image

def compute_kappa_shear_map_from_deflections(alpha_x: Image, 
                                             alpha_y: Image) -> tuple[Image, Image, Image, Image]:
    """
    This function computes the convergence and shear maps from the deflection fields alpha_x and alpha_y.
    The convergence and shear can be computed from the derivatives of the deflection fields.

    This function is based on Eq. 2.30, 3.38, 2.39 and 2.40 from "Introduction to Gravitational Lensing Lecture scripts" by
    Massimo Meneghetti (https://www.ita.uni-heidelberg.de/~jmerten/misc/meneghetti_lensing.pdf)

    This function is intended to be used with deflection fields computed at a given redshift 
    when no shear and convergence maps are available.

    Parameters
    ----------
    alpha_x, alpha_y : Image
        The deflection fields in the x and y directions, as MPDAF Image objects.

    Returns
    -------
    gamma_image : Image
        The computed shear map as an MPDAF Image object.
    gamma1_image : Image
        The computed gamma1 shear component map as an MPDAF Image object.
    gamma2_image : Image
        The computed gamma2 shear component map as an MPDAF Image object.
    kappa_image : Image
        The computed convergence map as an MPDAF Image object.
    """

    # pixel scale of the map
    dx = alpha_x.wcs.wcs.proj_plane_pixel_scales()[0].to_value('arcsec')
    dy = alpha_x.wcs.wcs.proj_plane_pixel_scales()[1].to_value('arcsec')

    # np.gradient returns derivatives in axis order:
    # axis 0 -> y direction
    # axis 1 -> x direction
    dax_dy, dax_dx = np.gradient(alpha_x.data, dy, dx)
    day_dy, day_dx = np.gradient(alpha_y.data, dy, dx)

    gamma1 = 0.5*(dax_dx - day_dy)
    gamma2 = dax_dx
    gamma = np.sqrt(gamma1**2 + gamma2**2)

    gamma_image = alpha_x.copy()
    gamma_image.data = gamma

    gamma1_image = alpha_x.copy()
    gamma1_image.data = gamma1

    gamma2_image = alpha_x.copy()
    gamma2_image.data = gamma2

    kappa = 0.5*(dax_dx + day_dy)
    kappa_image = alpha_x.copy()
    kappa_image.data = kappa

    return  gamma_image, gamma1_image, gamma2_image, kappa_image



# ============================================================
# Lensing utilities
# ============================================================
def make_delens(ra_array: np.ndarray, 
                dec_array: np.ndarray, 
                dir_matrices: str='.', 
                alpha_x_filename: str='alpha_x.fits', 
                alpha_y_filename: str='alpha_y.fits'):
    """
    Delens sky positions using the deflection matrices. It assumes that the deflection matrices are given in the same WCS as the input coordinates,
    and that the deflection values are in arcseconds.
    It also assumes that positve x deflections correspond to negative RA deflections, and positive y deflections correspond to positive Dec deflections, 
    which is the standard convention for LENSTOOL.
    It also assumes the deflections are normalized at the desired redshift configuration, so no scaling is applied to the deflection matrices.

    Parameters
    ----------
    ra_array, dec_array : array-like
        Arrays of RA and Dec coordinates to be delensed, in degrees.
    dir_matrices : str
        Directory where the deflection matrices are stored.
    alpha_x_filename, alpha_y_filename : str
        Filenames of the deflection matrices in the x and y directions, respectively. 
        The files should be in FITS format and contain the deflection values in arcseconds.
    
    Returns
    -------
    ra_abs, dec_abs : ndarray
        Delensed/source-plane coordinates in degrees.
    """

    alpha_x_path = os.path.join(dir_matrices, alpha_x_filename)
    alpha_y_path = os.path.join(dir_matrices, alpha_y_filename)

    alpha_x = Image(alpha_x_path)
    alpha_y = Image(alpha_y_path)

    wcs_1 = alpha_x.wcs  # alpha_x and alpha_y should share WCS

    # Make inputs arrays
    ra_array = np.atleast_1d(np.asarray(ra_array, dtype=float))
    dec_array = np.atleast_1d(np.asarray(dec_array, dtype=float))

    # Rotate the deflection field into RA/Dec-aligned components
    # in case the deflection matrices are not aligned with the RA/Dec axes (e.g., if they were computed in a rotated frame)
    rot = np.radians(-wcs_1.get_rot())

    ax0 = alpha_x.data.copy()
    ay0 = alpha_y.data.copy()

    # Rotate the deflection field into RA/Dec-aligned components
    ax_rot = ax0 * np.cos(rot) - ay0 * np.sin(rot)
    ay_rot = ax0 * np.sin(rot) + ay0 * np.cos(rot)

    alpha_x.data = ax_rot
    alpha_y.data = ay_rot

    decra_array = np.column_stack([dec_array, ra_array])
    yx_array = wcs_1.sky2pix(decra_array, nearest=True)

    y_array = yx_array[:, 0].astype(int)
    x_array = yx_array[:, 1].astype(int)

    # Clip indices to valid range
    ny, nx = alpha_x.data.shape
    y_array = np.clip(y_array, 0, ny - 1)
    x_array = np.clip(x_array, 0, nx - 1)

    # Calculate the delensed/source-plane coordinates by subtracting the deflections from the observed coordinates
    # the +/- signs are because of the convention that positive x deflections correspond to decreasing RA, and positive y to increasing Dec
    ra_abs = ra_array + alpha_x[y_array, x_array].data / 3600.0 / np.abs(np.cos(np.deg2rad(dec_array)))
    dec_abs = dec_array - alpha_y[y_array, x_array].data / 3600.0

    return np.asarray(ra_abs), np.asarray(dec_abs)



def get_critical_line_mask(mu_map, min_length_pix=300):
    """
    Compute a mask of the critical line from the magnification map by finding the zero-level contours of the inverse magnification map (1/mu).
    The function returns a boolean mask where pixels on the critical line are True, and the rest are False. 
    The function also filters the contours to keep only those that are longer than a given minimum

    Parameters
    ----------
    mu_map : Image
        The magnification map as an MPDAF Image object.
    min_length_pix : int
        Minimum length in pixels for a contour to be considered part of the critical line. This is
        used to filter out small contours that are likely due to noise. 
        The optimal value may depend on the resolution of the map and the expected size of the critical line features.
    
    Returns
    -------
    line_mask : 2D boolean array
        A boolean mask where pixels on the critical line are True, and the rest are False.
    """

    # The mask is created on the inverse magnification map, 
    # since the critical line corresponds to infinite magnification (mu -> infinity), which means 1/mu -> 0.
    inv_mu = 1 / mu_map.data

    # find zero-level contours
    contours = find_contours(inv_mu, level=0.0)
    contours_xy = [np.column_stack([c[:, 1], c[:, 0]]) for c in contours]
    large_contours = []

    for verts in contours_xy:
        if len(verts) < 100:
            continue

        seglen = np.sqrt(np.sum(np.diff(verts, axis=0)**2, axis=1))
        arc_length = np.sum(seglen)

        if arc_length < 50:  # tune this
            continue

        large_contours.append(verts)

    large_contours = []

    for verts in contours_xy:
        if len(verts) < 100:
            continue

        seglen = np.sqrt(np.sum(np.diff(verts, axis=0)**2, axis=1))
        arc_length = np.sum(seglen)

        if arc_length < min_length_pix:  # tune this
            continue

        large_contours.append(verts)

    def contours_to_mask(contours, shape, thickness=1):
        ny, nx = shape
        mask = np.zeros((ny, nx), dtype=bool)

        for verts in contours:
            for p0, p1 in zip(verts[:-1], verts[1:]):
                seg = p1 - p0
                length = np.hypot(seg[0], seg[1])
                n = max(2, int(np.ceil(length * 2)))

                t = np.linspace(0, 1, n)
                pts = p0[None, :] + t[:, None] * seg[None, :]

                x = np.clip(np.round(pts[:, 0]).astype(int), 0, nx-1)
                y = np.clip(np.round(pts[:, 1]).astype(int), 0, ny-1)

                mask[y, x] = True

        if thickness > 1:
            mask = binary_dilation(mask, iterations=thickness-1)

        return mask


    line_mask = contours_to_mask(large_contours, inv_mu.shape, thickness=1)
    return line_mask



def distance_to_critical_line_map(mu_map, wcs_xy, x_array, y_array):
    """
    Compute a map of the distance to the critical line for a set of input coordinates (x_array, y_array) in pixel space, given a magnification map and its WCS.
    The function first computes a mask of the critical line from the magnification map, 
    and then computes the distance from each input coordinate to the nearest pixel on the critical line, returning a dictionary with
    the distances for each input coordinate.

    Parameters
    ----------
    mu_map : Image
        The magnification map as an MPDAF Image object.
    wcs_xy : WCS
        The WCS object corresponding to the pixel coordinates of the magnification map.
    x_array, y_array : array-like
        Arrays of x and y pixel coordinates for which to compute the distance to the critical line.
    
    Returns
    -------
    distances : dict
        A dictionary where the keys are tuples of input coordinates (x, y) and the values are the distances from those coordinates to the nearest pixel on the critical line, in the same units as
        the pixel scale of the magnification map.
    line_mask : 2D boolean array
        A boolean mask where pixels on the critical line are True, and the rest are False.
        
    """

    inv_mu = 1 / mu_map.data

    
    line_mask = get_critical_line_mask(mu_map, min_length_pix=300)

    distance_map = distance_transform_edt(~line_mask)
    pix_coords = np.column_stack((x_array, y_array))
    world_coords = wcs_xy.wcs.wcs_pix2world(pix_coords, 0)
    x_mu, y_mu = mu_map.wcs.wcs.wcs_world2pix(world_coords, 0).T
    x_mu = x_mu.astype(int)
    y_mu = y_mu.astype(int)

    
    scale = mu_map.wcs.wcs.proj_plane_pixel_scales()[0].to_value(u.arcsec)

    distances = dict()

    for x, y, x_mu_value, y_mu_value in zip(x_array, y_array, x_mu, y_mu):
        distances[(x, y)] = distance_map[y_mu_value, x_mu_value]*scale

    return distances, line_mask
