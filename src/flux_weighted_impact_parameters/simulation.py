import numpy as np



def exponential_radial_profile(impact_parameter_map, scale_length, normalization, random_noise=0.1, is_noise_log=True):
    """
    Compute an exponential radial profile given an impact parameter map.

    Parameters
    ----------
    impact_parameter_map : 2D array
        A map of impact parameters (in kpc) for each pixel.
    scale_length : float
        The scale length of the exponential profile (in kpc).
    normalization : float
        The normalization factor for the profile (in units of cm^-2).
    random_noise : float
        The standard deviation of the random noise to be added to the profile. Default is 0.1.
        Noise is independent for each pixel.
    is_noise_log : bool
        If True, the random noise is added in log space. If False, it is added in linear space. Default is True.

    Returns
    -------
    2D array
        The computed exponential radial profile for each pixel. In units of cm^-2.
    """

    model_field = impact_parameter_map.copy()
    model_field.data = normalization * np.exp(-model_field.data / scale_length)
    
    if random_noise > 0:
        if is_noise_log:
            log_field = np.log10(model_field.data)
            noise = np.random.normal(0, random_noise, size=model_field.data.shape)
            log_field += noise
            model_field.data = 10**log_field
        else:
            model_field.data += np.random.normal(0, random_noise, size=model_field.data.shape)
    
    return model_field


def get_observed_value(model_field, flux_contribution_maps):
    """
    Compute the flux-weighted mean and standard deviation of the model field given the flux contribution maps.
    The function runs on a list of flux contribution maps, where each map corresponds to the contribution of a given spaxel to the pixels in the image.

    Parameters
    ----------
    model_field : 2D array
        The model field (e.g., impact parameter map) for each pixel.
    flux_contribution_maps : 3D array
        A 3D array where each slice along the first axis corresponds to the flux contribution map for a given spaxel. The shape should be (n_spaxels, ny, nx).    
    
    Returns
    -------
    mean_values : 1D array
        The flux-weighted mean value of the model field for each spaxel.
    std_values : 1D array
        The flux-weighted standard deviation of the model field for each spaxel.
    """

    field = np.asarray(model_field.data)
    weights = np.asarray(flux_contribution_maps)

    # denominator per spaxel
    wsum = np.nansum(weights, axis=(1, 2))

    # weighted mean
    mean_values = np.nansum(weights * field[None, :, :], axis=(1, 2)) / wsum

    # weighted variance
    diff2 = (field[None, :, :] - mean_values[:, None, None])**2
    var_values = np.nansum(weights * diff2, axis=(1, 2)) / wsum

    return mean_values, np.sqrt(var_values)


