import numpy as np



def compute_pixel_correlation_matrix(pixel_stats_dict):
    """
    Compute the cosine-similarity correlation matrix between spaxel
    contribution kernels.

    Returns
    -------
    corr : 2D ndarray
        Correlation matrix.
    labels : list of tuple
        List of (x, y) spaxel coordinates in the same order as corr.
    kernels : 2D ndarray
        Flattened normalized kernels, shape (n_spaxels, n_hst_pixels).
    """

    xx, yy = zip(*pixel_stats_dict.keys())

    labels = [(int(x), int(y)) for x, y in zip(xx, yy)]

    kernels = []

    for x, y in zip(xx, yy):
        stats = pixel_stats_dict[(x, y)]
        contrib_map = stats['contrib_map']
        contrib_map = contrib_map / np.sum(contrib_map) if np.sum(contrib_map) > 0 else contrib_map
        kernels.append(contrib_map.ravel())

    kernels = np.array(kernels, dtype=float)

    # protect against empty/zero kernels
    norm = np.sqrt(np.sum(kernels**2, axis=1))
    good = norm > 0

    kernels = kernels[good]
    labels = [lab for lab, keep in zip(labels, good) if keep]
    norm = norm[good]

    kernels_norm = kernels / norm[:, None]
    corr = kernels_norm @ kernels_norm.T

    return corr, labels, kernels_norm

def compute_normalized_correlations_matrix(pixel_stats_dict):
    xx, yy = zip(*pixel_stats_dict.keys())

    labels = [(int(x), int(y)) for x, y in zip(xx, yy)]

    kernels = []

    for x, y in zip(xx, yy):
        stats = pixel_stats_dict[(x, y)]
        contrib_map = stats['contrib_map']
        contrib_map = contrib_map / np.sum(contrib_map) 
        mean, std = np.mean(contrib_map), np.std(contrib_map)
        contrib_map = (contrib_map - mean) / std
        kernels.append(contrib_map.ravel())

    kernels = np.array(kernels, dtype=float)

    # protect against empty/zero kernels
    norm = np.sqrt(np.sum(kernels**2, axis=1))
    good = norm > 0

    kernels = kernels[good]
    labels = [lab for lab, keep in zip(labels, good) if keep]
    norm = norm[good]

    kernels_norm = kernels / norm[:, None]
    corr = kernels_norm @ kernels_norm.T

    return corr, labels, kernels_norm


def overlap_coefficients(pixel_stats_dict):
    xx, yy = zip(*pixel_stats_dict.keys())

    labels = [(int(x), int(y)) for x, y in zip(xx, yy)]

    kernels = []

    for x, y in zip(xx, yy):
        stats = pixel_stats_dict[(x, y)]
        contrib_map = stats['contrib_map']
        contrib_map = contrib_map / np.sum(contrib_map) 
        kernels.append(contrib_map.ravel())

    kernels = np.array(kernels, dtype=float)

    # protect against empty/zero kernels
    norm = np.sqrt(np.sum(kernels**2, axis=1))
    good = norm > 0

    kernels = kernels[good]
    labels = [lab for lab, keep in zip(labels, good) if keep]
    norm = norm[good]

    #kernels_norm = kernels / norm[:, None]
    kernels_norm = kernels 
    overlap_matrix = np.zeros((len(kernels_norm), len(kernels_norm)))

    for i in range(len(kernels_norm)):
        for j in range(i + 1):
            overlap_matrix[i, j] = np.sum(np.minimum(kernels_norm[i], kernels_norm[j]))
            overlap_matrix[j, i] = overlap_matrix[i, j]

    return overlap_matrix, labels, kernels_norm



def effective_number_of_independent_pixels(corr_matrix):
    """
    Compute the effective number of independent pixels contributing to the spaxel.

    Parameters
    ----------
    contrib_map : 2D array
        Contribution map for a given spaxel.

    Returns
    -------
    N_eff : float
        Effective number of independent pixels.
    """
    trR = np.trace(corr_matrix)
    trR2 = np.trace(np.matmul(corr_matrix, corr_matrix))
    return trR**2 / trR2 if trR2 > 0 else 0