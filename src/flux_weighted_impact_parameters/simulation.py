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
        The normalization factor for the profile.

    Returns
    -------
    2D array
        The computed exponential radial profile for each pixel.
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


