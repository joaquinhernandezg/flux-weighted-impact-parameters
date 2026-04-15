from astropy.cosmology import FlatLambdaCDM
# force reimport
from mpdaf.obj import Image
import astropy.units as u
import numpy as np
import os
from scipy.ndimage import distance_transform_edt


def scale_lensing_matrix(lensing_matrix_filename, z_lens, z_source, z_norm_matrices, output_filename):
    """
    Scale the lensing matrix from the normalization redshift to the desired lens and source redshifts.

    param lensing_matrix_filename: str, the filename of the lensing matrix to be scaled
    param z_lens: float, the redshift of the lens
    param z_source: float, the redshift of the source
    param z_norm_matrices: float, the redshift at which the lensing matrix was originally computed
    param output_filename: str, the filename to save the scaled lensing matrix to
    return: the scaled lensing matrix as a numpy array
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


def compute_magnification_map_from_deflections(alpha_x, alpha_y, save_filename=None):
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

def compute_kappa_shear_map_from_deflections(alpha_x, alpha_y):
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
def make_delens(ra_array, dec_array, dir_matrices='.', alpha_x_filename='alpha_x.fits', alpha_y_filename='alpha_y.fits'):
    """
    Delens sky positions using the deflection matrices.

    Parameters
    ----------
    ra_array, dec_array : float or array-like
        Sky coordinates in degrees.

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
    rot = np.radians(-wcs_1.get_rot())

    ax0 = alpha_x.data.copy()
    ay0 = alpha_y.data.copy()

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

    # Deflections are in arcsec
    ra_abs = ra_array + alpha_x[y_array, x_array].data / 3600.0 / np.abs(np.cos(np.deg2rad(dec_array)))
    dec_abs = dec_array - alpha_y[y_array, x_array].data / 3600.0

    return np.asarray(ra_abs), np.asarray(dec_abs)

def get_critical_line_mask(mu_map, min_length_pix=300):
    from skimage.measure import find_contours

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


    from scipy.ndimage import binary_dilation

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
    from skimage.measure import find_contours

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
