import numpy as np
from flux_weighted_impact_parameters import covariance_matrix, covariance_matrix_fast
from flux_weighted_impact_parameters import get_observed_value, exponential_radial_profile
from scipy.linalg import cho_factor, cho_solve



def log_likelihood(params, y, y_err, impact_parameter_map, 
                   flux_contribution_maps,
                   geometrical_covariance_matrix):
    
    scale_length, normalization, random_noise = params
    model_field_sim = np.log10(exponential_radial_profile(impact_parameter_map,
                                                        scale_length, normalization, random_noise=0).data)
    m, __import__ = get_observed_value(model_field_sim, flux_contribution_maps)

    Sigma = covariance_matrix_fast(geometrical_covariance_matrix, y_err, random_noise)

    r = y - m

    # Cholesky for stability
    try:
        c, lower = cho_factor(Sigma)
    except np.linalg.LinAlgError:
        return -np.inf  # reject non-positive definite

    r = y - m

    # quadratic term
    chi2 = r @ cho_solve((c, lower), r)

    # log determinant
    logdet = 2 * np.sum(np.log(np.diag(c)))

    n = len(y)

    return -0.5 * (chi2 + logdet + n * np.log(2*np.pi))

    