from .impact_parameter_distribution import (
    azimuthal_angle_stats_for_binned_pixel,
    impact_parameter_stats_for_binned_pixel,
)
from .flux_contribution_maps import compute_input_contribution_map_for_binned_pixel
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import numpy as np
from .lensing import distance_to_critical_line_map
from .rebinning import make_pixel_geometric_mask
from mpdaf.obj import Image


def get_binned_pixel_stats(highres_image: Image, 
                            arc_mask_highres: Image, 
                            rebinned_image: Image, 
                            rebinned_mask: Image, 
                            highres_psf_fwhm: float, 
                            lowres_psf_fwhm: float, 
                            magnification_map: Image,
                            impact_parameter_map: Image, 
                            az_angle_map: Image, 
                            shear1_map: Image, 
                            shear2_map: Image, 
                            kappa_map: Image,
                            max_workers=None,
                            output_dir=f'../plots/flux_contribution_maps'):

    yy, xx = np.indices(rebinned_image.data.shape)
    yy, xx = yy.astype(int), xx.astype(int)
    xx, yy = xx[rebinned_mask.data > 0], yy[rebinned_mask.data > 0]

    binned_pixel_stats = {}
    os.makedirs(output_dir, exist_ok=True)

    distances, line_mask = distance_to_critical_line_map(magnification_map, 
                                              wcs_xy=rebinned_image.wcs, x_array=xx, y_array=yy)
    shear_map = np.sqrt(shear1_map.data**2 + shear2_map.data**2)


    def process_pixel(x, y):
        contrib_map, weighted_flux_map, total_flux, sens_map, footprint_img = compute_input_contribution_map_for_binned_pixel(
            highres_image=highres_image,
            lowres_image=rebinned_image,
            highres_psf_FWHM=highres_psf_fwhm,
            lowres_psf_FWHM=lowres_psf_fwhm,
            x_pix_lowres=x,
            y_pix_lowres=y,
            arc_mask_highres=arc_mask_highres)


        impact_parameter_stats = impact_parameter_stats_for_binned_pixel(
            highres_image=highres_image,
            lowres_image=rebinned_image,
            highres_psf_FWHM=highres_psf_fwhm,
            lowres_psf_FWHM=lowres_psf_fwhm,
            x_pix_lowres=x,
            y_pix_lowres=y,
            arc_mask_highres=arc_mask_highres,
            impact_parameter_map=impact_parameter_map.data
        )
        az_stats = azimuthal_angle_stats_for_binned_pixel(
            highres_image=highres_image,
            lowres_image=rebinned_image,
            highres_psf_FWHM=highres_psf_fwhm,
            lowres_psf_FWHM=lowres_psf_fwhm,
            x_pix_lowres=x,
            y_pix_lowres=y,
            arc_mask_highres=arc_mask_highres,
            azimuthal_angle_map=az_angle_map.data
        )

        #weighted_coefficient_of_variation
        w = weighted_flux_map
        f = highres_image.data
        mu_f = np.nansum(w*f)
        sigma_F_2 = np.nansum(w*(f-mu_f)**2)
        CV_F = np.sqrt(sigma_F_2)/mu_f

        # effective number of contributing pixels
        p = w*f/(np.sum(w*f))
        N_eff = 1/np.nansum(p**2)
        p_1 = np.nanpercentile(w, 1)
        N_supp = np.sum(w>=p_1 )
        f_eff = N_eff/N_supp

        # entropy
        entropy =0

        flux_no_conv = np.sum(highres_image.data*footprint_img/np.sum(footprint_img)*arc_mask_highres.data)
        flux_conv = np.sum(highres_image.data*sens_map/np.sum(sens_map)*arc_mask_highres.data)
        psf_smeared_fraction = flux_no_conv/flux_conv


        total_flux_spaxel = np.sum(highres_image.data*sens_map/np.sum(sens_map))
        total_flux_arc = np.sum(highres_image.data*sens_map/np.sum(sens_map)*arc_mask_highres.data)

        smeared_flux_fraction = total_flux_arc/total_flux_spaxel


 
        #mean_shear = np.sum(contrib_map*shear_map.data)/np.sum(contrib_map)
        #mean_shear1 = np.sum(contrib_map*shear1_map.data)/np.sum(contrib_map)
        #mean_shear2 = np.sum(contrib_map*shear2_map.data)/np.sum(contrib_map)
        #mean_kappa = np.sum(contrib_map*kappa_map.data)/np.sum(contrib_map)
        #ut = 1/(1-kappa_map.data-shear_map.data)
        #ur = 1/(1-kappa_map.data+shear_map.data)
        #mean_ut = np.sum(contrib_map*ut)/np.sum(contrib_map)
        #mean_ur = np.sum(contrib_map*ur)/np.sum(contrib_map)
        #phi_gamma = 0.5*np.arctan2(shear2_map.data, shear1_map.data)
        #mean_phi_gamma = np.sum(contrib_map*phi_gamma)/np.sum(contrib_map)

        mean_magnification = np.sum(contrib_map*magnification_map.data)/np.sum(contrib_map)
        
    

        pixel_stats = {
            "contrib_map": contrib_map,
            "weighted_flux_map": weighted_flux_map,
            "total_flux": total_flux,
            "impact_parameter_stats": impact_parameter_stats,
            "azimuthal_angle_stats": az_stats,
            "sensitivity_map": sens_map,
            "footprint_img": footprint_img,
            "distance_to_critical_line": distances[(x, y)],
            "local_magnification": mean_magnification,
            "weighted_coefficient_of_variation": CV_F,
            "effective_flux_contributing_pixels_fraction": f_eff,   
            "psf_smeared_fraction": smeared_flux_fraction,
            #"entropy": entropy,
            #"mean_gamma": mean_shear,
            #"mean_gamma1": mean_shear1,
            #"mean_gamma2": mean_shear2,
            #"mean_kappa": mean_kappa,
            #"mean_ut": mean_ut,
            #"mean_ur": mean_ur,
            #"mean_phi_gamma": mean_phi_gamma

        }
        return (x, y), pixel_stats

    for x, y in zip(xx, yy):
        print(f"Processing pixel (x={x}, y={y})...")
        pixel_key, pixel_stats = process_pixel(x, y)
        binned_pixel_stats[pixel_key] = pixel_stats


    return binned_pixel_stats
