from collections import defaultdict
from astropy.table import Table


def make_summary_table_spaxels(binned_pixel_stats):
    spaxels_table = defaultdict(list)

    for x, y in binned_pixel_stats.keys():
        spaxel_stats = binned_pixel_stats[(x, y)]
        impact_parameter_stats = spaxel_stats["impact_parameter_stats"]

        spaxels_table["x_bin"].append(x)
        spaxels_table["y_bin"].append(y)

        flux_weighted = impact_parameter_stats['mean']
        flux_weighted_error = impact_parameter_stats['std']
        impact_parameter_centre = impact_parameter_stats['impact_parameter_centre']
        impact_parameter_centre_error = impact_parameter_stats['impact_parameter_centre_error']
        local_gradiet_impact_parameter = impact_parameter_stats['local_gradient_magnitude']

        
        spaxels_table["D_flux_weighted"].append(flux_weighted)
        spaxels_table["D_flux_weighted_error"].append(flux_weighted_error)
        spaxels_table["D_centre"].append(impact_parameter_centre)
        spaxels_table["D_centre_error"].append(impact_parameter_centre_error)
        spaxels_table["local_gradient_impact_parameter"].append(local_gradiet_impact_parameter)
        spaxels_table['D_relative_error'].append((flux_weighted-impact_parameter_centre)/flux_weighted)
    
        az_angle_stats = spaxel_stats["azimuthal_angle_stats"]
        flux_weighted = az_angle_stats['mean']
        flux_weighted_error = az_angle_stats['std']
        centre = az_angle_stats['azimuthal_angle_centre']
        centre_error = az_angle_stats['azimuthal_angle_centre_error']
        local_gradient_azimuthal_angle = az_angle_stats['local_gradient_magnitude']

        spaxels_table["az_flux_weighted"].append(flux_weighted)
        spaxels_table["az_flux_weighted_error"].append(flux_weighted_error)
        spaxels_table["az_centre"].append(centre)
        spaxels_table["az_centre_error"].append(centre_error)
        spaxels_table["local_gradient_azimuthal_angle"].append(local_gradient_azimuthal_angle)


        local_gradient_magnification = spaxel_stats['local_magnification']
        spaxels_table["local_magnification"].append(local_gradient_magnification)



        f_eff = spaxel_stats["weighted_coefficient_of_variation"]
        CV_f = spaxel_stats["effective_flux_contributing_pixels_fraction"]
        d_critical_line = spaxel_stats["distance_to_critical_line"]

        spaxels_table["CV_f"].append(CV_f)
        spaxels_table["f_eff"].append(f_eff)
        spaxels_table["d_critical_line"].append(d_critical_line)
        spaxels_table["psf_smeared_fraction"].append(spaxel_stats["psf_smeared_fraction"])

        

    return Table(spaxels_table)
