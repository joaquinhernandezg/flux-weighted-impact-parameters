import matplotlib.pyplot as plt
from .rebinning import get_extent, get_extent_in_other_frame
from astropy.stats import sigma_clipped_stats
import numpy as np
import astropy.units as u
from matplotlib.patches import ConnectionPatch
import matplotlib.patches as patches



def plot_HST_and_MUSE_cutouts(hst_cutout, muse_cutout_oversampled, out_filename="HST_MNUSE_cutouts.png"):
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))

    extent_hst = get_extent(hst_cutout)
    extent_muse = get_extent(muse_cutout_oversampled)
    ax[0].imshow(hst_cutout.data, origin='lower', cmap='gray', extent=extent_hst)
    ax[0].set_title('HST Cutout')
    ax[0].set_xlabel('HST x [pix]')
    ax[0].set_ylabel('HST y [pix]')

    ax[1].imshow(muse_cutout_oversampled.data, origin='lower', cmap='gray', extent=extent_muse)
    ax[1].set_title('MUSE White Image (oversampled to HST res)')
    ax[1].set_xlabel('HST x [pix]')
    ax[1].set_ylabel('HST y [pix]')

    plt.tight_layout()
    fig.savefig(out_filename)

def plot_HST_original_convolved_rebinned_and_mask(hst_cutout, hst_convolved, hst_rebinned, mask_rebinned_4x4, out_filename="HST_convolved_rebinned_mask.png"):
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    extent_hst = get_extent(hst_cutout)
    extent_muse = get_extent(hst_rebinned)
    extent_muse_on_hst = get_extent_in_other_frame(hst_rebinned, hst_cutout)


    ax[0, 0].imshow(hst_cutout.data, origin='lower', cmap='gray', extent=extent_hst)
    ax[0, 0].set_title('HST Cutout')
    ax[0, 0].set_xlabel('HST x [pix]')
    ax[0, 0].set_ylabel('HST y [pix]')

    ax[0, 1].imshow(hst_convolved.data, origin='lower', cmap='gray', extent=extent_hst)
    ax[0, 1].set_title('HST Convolved to MUSE PSF')
    ax[0, 1].set_xlabel('HST x [pix]')
    ax[0, 1].set_ylabel('HST y [pix]')

    ax[1, 0].imshow(hst_rebinned.data, origin='lower', cmap='gray', extent=extent_muse)
    ax[1, 0].set_title('HST Convolved and Rebinned to MUSE Pixel Scale')
    ax[1, 0].set_xlabel('MUSE x [pix]')
    ax[1, 0].set_ylabel('MUSE y [pix]')

    ax[1, 1].imshow(mask_rebinned_4x4.data, origin='lower', cmap='gray', extent=extent_muse_on_hst)
    ax[1, 1].set_title('Mask Rebinned to MUSE Pixel Scale')
    ax[1, 1].set_xlabel('MUSE x [pix]')
    ax[1, 1].set_ylabel('MUSE y [pix]')
    
    plt.tight_layout()
    fig.savefig(out_filename)


def plot_spaxels_on_highres(hst_cutout, mask_rebinned, filename="HST_cutout_with_masked_spaxels.png"):
    extent_hst = get_extent(hst_cutout)
    extent_muse_on_hst = get_extent_in_other_frame(mask_rebinned, hst_cutout)
    extent_hst_on_muse = get_extent_in_other_frame(hst_cutout, mask_rebinned)

    # now for each xx, yy in the mask, draw the corresponding square in the high resolution image and compute the mean flux in that square, and assign that mean flux to the corresponding pixel in the rebinned image
    fig, ax = plt.subplots(figsize=(12, 12))
    mean, median, std = sigma_clipped_stats(hst_cutout.data, sigma=3)
    ax.imshow(hst_cutout.data, origin='lower', cmap='gray', vmin=-std, vmax=mean+5*std, extent=extent_hst_on_muse)
    ax.set_title('HST Cutout with Masked Regions')
    ax.set_xlabel('RA')
    ax.set_ylabel('DEC')
    yy_bin, xx_bin = np.indices(mask_rebinned.data.shape)
    yy_bin = yy_bin[mask_rebinned.data > 0].flatten()
    xx_bin = xx_bin[mask_rebinned.data > 0].flatten()


    for x, y in zip(xx_bin, yy_bin):
        rect = plt.Rectangle(((x-0.5), (y-0.5)), 1, 1, edgecolor='red', facecolor='none', linewidth=1, alpha=1)
        ax.add_patch(rect)
        ax.text((x+0.5), (y+0.5), f"({x},\n {y})", color='w', fontsize=5, ha='center', va='center')

    # now make an index array for the mask
    yy,xx = np.indices(mask_rebinned.data.shape)
    yy, xx = (yy.flatten()), (xx.flatten())
    ax.scatter(xx, yy, s=1, color='w', alpha=0.5, label='Spaxel centres')

    plt.tight_layout()
    fig.savefig(filename)

def plot_spaxels_on_lowres(hst_binned, mask_rebinned, filename="HST_cutout_with_masked_spaxels.png"):
    # now for each xx, yy in the mask, draw the corresponding square in the high resolution image and compute the mean flux in that square, and assign that mean flux to the corresponding pixel in the rebinned image
    fig, ax = plt.subplots(figsize=(12, 12))
    mean, median, std = sigma_clipped_stats(hst_binned.data, sigma=3)
    ax.imshow(hst_binned.data, origin='lower', cmap='gray', vmin=-std, vmax=mean+5*std)
    ax.set_title('HST Binned with Masked Regions')
    ax.set_xlabel('RA')
    ax.set_ylabel('DEC')
    yy_bin, xx_bin = np.indices(mask_rebinned.data.shape)
    yy_bin = yy_bin[mask_rebinned.data > 0].flatten()
    xx_bin = xx_bin[mask_rebinned.data > 0].flatten()


    for x, y in zip(xx_bin, yy_bin):
        rect = plt.Rectangle(((x-0.5), (y-0.5)), 1, 1, edgecolor='red', facecolor='none', linewidth=1, alpha=1)
        ax.add_patch(rect)
        ax.text((x)*1, (y)*1, f"({x},\n {y})", color='w', fontsize=5, ha='center', va='center')

    # now make an index array for the mask
    yy,xx = np.indices(mask_rebinned.data.shape)
    yy, xx = (yy.flatten())*1, (xx.flatten())*1
    ax.scatter(xx, yy, s=1, color='w', alpha=0.5, label='Spaxel centres')

    plt.tight_layout()
    fig.savefig(filename)


def plot_example_contribution_map_for_binned_pixel(x_bin, y_bin, binned_pixel_stats, hst_rebinned, hst_cutout, out_filename="example_contribution_map.png",
                                                   ):
    # now do a plot of the high resolution image, draw the spaxel 10, 20 of the low resolution image with a rectangle. In the second panel show the convolved mask. In the third panel show the contribution map for that spaxel, and in the fourth panel show the impact parameter distribution for that spaxel with the centre and error shaded region.

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    binned_pixel_stats_spaxel = binned_pixel_stats[(x_bin, y_bin)]

    # compute extents
    extent_hst = get_extent(hst_cutout)
    extent_lowres = get_extent(hst_rebinned)
    extent_lowres_on_hst = get_extent_in_other_frame(hst_rebinned, hst_cutout)

    # derive highres-per-lowres scale from extents (in highres pixels)
    scale_highres_per_lowres = (extent_lowres_on_hst[1] - extent_lowres_on_hst[0]) / float(hst_rebinned.data.shape[1])

    # panel 1: show the rebinned (low-res) image plotted in high-res coordinates
    mean, median, std = sigma_clipped_stats(hst_rebinned.data, sigma=3)
    axes[0].imshow(hst_rebinned.data, origin='lower', cmap='gray', vmin=-0.5*std, vmax=mean+3*std, extent=extent_lowres_on_hst)
    pad = 0.15 * scale_highres_per_lowres
    rect = plt.Rectangle((extent_lowres_on_hst[0] + x_bin * scale_highres_per_lowres, extent_lowres_on_hst[2] + y_bin * scale_highres_per_lowres), scale_highres_per_lowres, scale_highres_per_lowres, edgecolor='red', facecolor='none', linewidth=1)
    axes[0].add_patch(rect)
    axes[0].set_title('PSF-convolved low-resolution HST Image (displayed in HST coords)')
    axes[0].set_xlabel('HST x [pix]')
    axes[0].set_ylabel('HST y [pix]')

    axes[1].imshow(binned_pixel_stats_spaxel["sensitivity_map"], origin='lower', cmap='magma', vmin=1e-9)
    rect = plt.Rectangle((extent_lowres_on_hst[0] + x_bin * scale_highres_per_lowres, extent_lowres_on_hst[2] + y_bin * scale_highres_per_lowres), scale_highres_per_lowres, scale_highres_per_lowres, edgecolor='red', facecolor='none', linewidth=1)
    axes[1].add_patch(rect)
    axes[1].set_title('Sensitivity Map)')
    axes[1].set_xlabel('HST x [pix]')
    axes[1].set_ylabel('HST y [pix]')

    # set pixels equal to zero in the contribution map to alpha tranparent
    contrib = binned_pixel_stats_spaxel["contrib_map"].copy()

    # Mask negligible values so they are fully transparent
    threshold = 1e-10 * np.nanmax(contrib)
    contrib_masked = np.ma.masked_less_equal(contrib, threshold)


    axes[2].imshow(hst_cutout.data, origin='lower', cmap='gray', vmin=-std, vmax=mean+100*std, extent=extent_hst)
    axes[2].imshow(contrib_masked, origin='lower', cmap='magma', alpha=0.7, vmin=0, vmax=np.nanmax(contrib_masked), extent=extent_lowres_on_hst)
    rect = plt.Rectangle((extent_lowres_on_hst[0] + x_bin * scale_highres_per_lowres, extent_lowres_on_hst[2] + y_bin * scale_highres_per_lowres), scale_highres_per_lowres, scale_highres_per_lowres, edgecolor='red', facecolor='none', linewidth=1, alpha=1)
    axes[2].add_patch(rect)


    axes[2].set_title('Flux Contribution Map')
    axes[2].set_xlabel('HST x [pix]')
    axes[2].set_ylabel('HST y [pix]')

    ax = axes[3]
    impact_parameter_stats = binned_pixel_stats_spaxel["impact_parameter_stats"]
    impact_parameter_centre = impact_parameter_stats['impact_parameter_centre']
    impact_parameter_centre_error = impact_parameter_stats['impact_parameter_centre_error']
    ax.hist(impact_parameter_stats['values'], bins=np.linspace(0, 30, 30), histtype='step', color='blue', density=True, label='Flux weighted distribution')
    ax.axvline(impact_parameter_centre, color='red', linestyle='--', label=f'Spaxel centre')
    ax.fill_betweenx([0, ax.get_ylim()[1]], impact_parameter_centre - impact_parameter_centre_error, impact_parameter_centre + impact_parameter_centre_error, color='red', alpha=0.3)
    ax.set_xlabel('Impact Parameter [kpc]')
    ax.set_ylabel('Weighted PDF')
    ax.set_title(f'Impact Parameter Distribution')
    ax.legend(loc='upper right')

    # zoom in on each axes according to the portion of the rectangle position
    n_width = 5
    half_span = (n_width + 0.5) * scale_highres_per_lowres
    center_x = extent_lowres_on_hst[0] + x_bin * scale_highres_per_lowres + 0.5 * scale_highres_per_lowres
    center_y = extent_lowres_on_hst[2] + y_bin * scale_highres_per_lowres + 0.5 * scale_highres_per_lowres
    axes[0].set_xlim(center_x - half_span, center_x + half_span)
    axes[0].set_ylim(center_y - half_span, center_y + half_span)
    axes[1].set_xlim(center_x - half_span - n_width, center_x + half_span + n_width)
    axes[1].set_ylim(center_y - half_span - n_width, center_y + half_span + n_width)
    axes[2].set_xlim(center_x - half_span - n_width, center_x + half_span + n_width)
    axes[2].set_ylim(center_y - half_span - n_width, center_y + half_span + n_width)

    
    fig.tight_layout()
    fig.savefig(out_filename, dpi=300)


def plot_scale_bar(ax, length_pix, start_x_frac, start_y_frac, text=None, color='white', fontsize=12):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    start_x = xlim[0] + start_x_frac * (xlim[1] - xlim[0])
    start_y = ylim[0] + start_y_frac * (ylim[1] - ylim[0])
    ax.plot([start_x, start_x + length_pix], [start_y, start_y], color=color, linewidth=2)
    if text is not None:
        ax.text(start_x + length_pix / 2, start_y - 0.02 * (ylim[1] - ylim[0]), text, color=color, fontsize=fontsize, ha='center', va='top')



def plot_hst_binned_pixels_contribution_map_and_impact_parameter_distribution(x_bin, y_bin, binned_pixel_stats, 
                                                                              hst_rebinned, hst_cutout, mask_arc_highres,
                                                                               out_filename="plot_hst_binned_pixels_contribution_map_and_impact_parameter_distribution.png",
                                                                            min_impact_parameter=0, max_impact_parameter=30, delta_bin=0.5):

    extent_highres_on_lowres = get_extent_in_other_frame(hst_cutout, hst_rebinned)
    extent_lowres = get_extent(hst_rebinned)
    xx_bin, yy_bin = zip(*binned_pixel_stats.keys())



    fig, axs = plt.subplots(1, 4, figsize=(16, 6))



    #============================================================
    # Panel 1: Show the highres image, but in lowres coordinates
    #============================================================
    ax = axs[0]
    # first plot hst image high revolution
    vmin, vmax = np.percentile(hst_cutout.data, [5, 99])
    im = ax.imshow(hst_cutout.data, origin='lower', cmap='gray', vmin=vmin, vmax=vmax, extent=extent_highres_on_lowres)
    # draw the arc mask and highlight G1    
    ax.contour(mask_arc_highres, origin='lower', colors='red', linewidths=0.5, levels=[0.5], extent=extent_highres_on_lowres)
    ax.set_xticks([])
    ax.set_yticks([])

    #============================================================
    # Panel 2: Show the lowres image, but in lowres coordinates with spaxels drawn in top
    #============================================================
    ax = axs[1]
    vmin, vmax = np.percentile(hst_rebinned.data, [5, 95])
    im = ax.imshow(hst_rebinned.data, origin='lower', cmap='gray', vmin=vmin, vmax=vmax, extent=extent_lowres)
    for xx, yy in zip(xx_bin, yy_bin):
        rect = plt.Rectangle(((xx-0.5), (yy-0.5)), 1, 1, edgecolor='red', facecolor='none', linewidth=0.5)
        ax.add_patch(rect)

    binned_pixel_scale = hst_rebinned.wcs.wcs.proj_plane_pixel_scales()[0].to_value(u.arcsec)


    # highlight the spaxel 13, 9 with a cyan rectangle
    rect = plt.Rectangle(((x_bin-0.5), (y_bin-0.5)), 1, 1, edgecolor='cyan', facecolor='none', linewidth=1)
    # highlight the square from the next panel
    rect2 = plt.Rectangle(((x_bin-2.5), (y_bin-2.5)), 5, 5, edgecolor='white', facecolor='none', linewidth=1, linestyle='--')
    ax.add_patch(rect)
    ax.add_patch(rect2)
    ax.set_xticks([])
    ax.set_yticks([])




    ax = axs[2]
    # plot flux contribution map for spaxel 13, 9
    # extent of MUSE binned spaxels
    binned_pixel_stats_spaxel = binned_pixel_stats[(x_bin, y_bin)]
    contrib = binned_pixel_stats_spaxel["contrib_map"].copy()/np.nanmax(binned_pixel_stats_spaxel["contrib_map"])
    # use extents for lowres image
    im = ax.imshow(contrib, origin='lower', cmap='magma', vmin=0, vmax=np.nanmax(contrib), extent=extent_highres_on_lowres)
    # draw spaxel
    rect = plt.Rectangle(((x_bin-0.5), (y_bin-0.5)), 1, 1, edgecolor='cyan', facecolor='none', linewidth=1)
    ax.add_patch(rect)

    mean, median, std = sigma_clipped_stats(hst_cutout.data, sigma=3)
    ax.contour(hst_cutout.data, origin='lower', colors='white', linewidths=0.2, levels=[5*std, 10*std, 30*std], extent=extent_highres_on_lowres)
    ax.set_xlim(x_bin-2.5, x_bin+2.5)
    ax.set_ylim(y_bin-2.5, y_bin+2.5)
    # remove a and y axis ticks
    ax.set_xticks([])
    ax.set_yticks([])


    cbar_ax = ax.inset_axes([0.4, 0.8, 0.5, 0.03])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.ax.set_title(r'w(p) [arbitrary units]', fontsize=10, color='white')
    cbar.ax.tick_params(labelsize=10, color='white', labelcolor='white')
    background = patches.Rectangle((-0.12, -2.5), 1.17, 5.5, 
                                transform=cbar_ax.transAxes, 
                                facecolor='grey', 
                                alpha=0.5, 
                                zorder=-1, 
                                clip_on=False)
    cbar_ax.add_patch(background)


    # --- Define zoom box in ax[1] ---
    x0, x1 = x_bin-2.5, x_bin+2.5
    y0, y1 = y_bin-2.5, y_bin+2.5

    # corners in ax[1]
    corners_ax1 = [
        (x1, y1),  # bottom-left
        (x1, y0),  # bottom-right
    ]

    # --- Define corresponding corners in ax[2] (its visible limits) ---
    xlim = axs[2].get_xlim()
    ylim = axs[2].get_ylim()

    corners_ax2 = [
        (xlim[0], ylim[1]),
        (xlim[0], ylim[0]),
    ]

    # --- Draw connections ---
    for (xyA, xyB) in zip(corners_ax1, corners_ax2):
        con = ConnectionPatch(
            xyA=xyA, coordsA=axs[1].transData,
            xyB=xyB, coordsB=axs[2].transData,
            color="white", linewidth=1.0
        )
        fig.add_artist(con)



    # plot the distribution of impact parameters, and weight them by the flux contribution, for the spaxel 13, 9
    ax = axs[3]
    impact_parameter_stats = binned_pixel_stats_spaxel["impact_parameter_stats"]
    impact_parameter_centre = impact_parameter_stats['impact_parameter_centre']
    impact_parameter_centre_error = impact_parameter_stats['impact_parameter_centre_error']
    mean_impact_parameters = np.sum(impact_parameter_stats['weights']*impact_parameter_stats['values'])/np.sum(impact_parameter_stats['weights'])
    std_impact_parameters = np.sqrt(np.sum(impact_parameter_stats['weights']*(impact_parameter_stats['values']-mean_impact_parameters)**2)/np.sum(impact_parameter_stats['weights']))

    # make histogram with numpy
    bins = np.arange(min_impact_parameter, max_impact_parameter, 0.2)
    hist, bin_edges = np.histogram(impact_parameter_stats['values'], bins=bins, weights=impact_parameter_stats['weights'], density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # histogram like only edges of the bars
    ax.bar(bin_centers, hist, width=bin_edges[1]-bin_edges[0], alpha=0.7, edgecolor='black', label='Flux-weighted distribution', color='tab:blue')

    hist, bin_edges = np.histogram(impact_parameter_stats['geometric_values'], bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.bar(bin_centers, hist, width=bin_edges[1]-bin_edges[0], alpha=0.7, edgecolor='black', label='Geometric distribution', color='tab:red')
    std_center = np.std(impact_parameter_stats['geometric_values'])



    ax.errorbar(mean_impact_parameters, 0.6, xerr=std_impact_parameters, fmt='o', color='tab:blue', label=f'Flux-weighted')
    ax.errorbar(impact_parameter_centre, 0.38, xerr=impact_parameter_centre_error, fmt='o', color='tab:red', label='Spaxel centre')


    ax.legend(loc='upper right')
    # move y axis to the right
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_xlabel('Impact Parameter [kpc]')
    ax.set_ylabel('Weighted PDF')
    # set axis aspect ratio to 1
    ax.set_box_aspect(1)
    ax.text(0.05, 0.9, 'd)', transform=ax.transAxes, fontsize=16, color='black', weight='bold')
    fig.tight_layout()

    fig.savefig(out_filename, dpi=300, bbox_inches='tight')
    return fig, axs


def plot_displacement_maps(binned_pixel_stats, hst_cutout, hst_rebinned, out_filename="plot_displacement_maps.pdf"):
    extent_lowres_on_hst = get_extent_in_other_frame(hst_cutout, hst_rebinned)
    extent_highres_on_lowres = get_extent_in_other_frame(hst_rebinned, hst_cutout)
    extent_highres = get_extent(hst_cutout)
    hst_pixscale = hst_cutout.wcs.get_axis_increments(u.arcsec)[0]
    scale_highres_per_lowres = (extent_highres_on_lowres[1] - extent_highres_on_lowres[0]) / float(hst_rebinned.data.shape[1])

    xx_bin, yy_bin = zip(*binned_pixel_stats.keys())

    # Collect all arrow data first (to normalize lengths)
    arrows = []
    for x_bin, y_bin in zip(xx_bin, yy_bin):
        spaxel_stats = binned_pixel_stats[(x_bin, y_bin)]
        contrib_map = spaxel_stats["contrib_map"]
        yy_hr, xx_hr = np.indices(contrib_map.shape)
        total_contrib = np.sum(contrib_map)

        if total_contrib > 0:
            x_weighted = np.sum(xx_hr * contrib_map) / total_contrib
            y_weighted = np.sum(yy_hr * contrib_map) / total_contrib

            x_spaxel_center = extent_lowres_on_hst[0] + x_bin * scale_highres_per_lowres + 0.5 * scale_highres_per_lowres
            y_spaxel_center = extent_lowres_on_hst[2] + y_bin * scale_highres_per_lowres + 0.5 * scale_highres_per_lowres

            dx = x_weighted - x_spaxel_center
            dy = y_weighted - y_spaxel_center
            length = np.sqrt(dx**2 + dy**2)
            arrows.append((x_spaxel_center, y_spaxel_center, dx, dy, length, x_bin, y_bin))

    # Normalize lengths for colormap
    lengths = np.array([a[4] for a in arrows])
    norm = plt.Normalize(vmin=lengths.min()*hst_pixscale, vmax=lengths.max()*hst_pixscale)
    cmap = plt.cm.plasma

    # Create figure
    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(16, 8))
    mean, median, std = sigma_clipped_stats(hst_cutout.data, sigma=3)
    ax = axs[0]
    ax.imshow(hst_cutout.data, origin='lower', cmap='gray', vmin=-std, vmax=mean + 20 * std, extent=extent_highres)

    for x_spaxel_center, y_spaxel_center, dx, dy, length, x_bin, y_bin in arrows:
        color = cmap(norm(length*hst_pixscale))

        pad = 0.01 * scale_highres_per_lowres
        left = extent_lowres_on_hst[0] + x_bin * scale_highres_per_lowres + pad
        bottom = extent_lowres_on_hst[2] + y_bin * scale_highres_per_lowres + pad
        side = scale_highres_per_lowres - 2 * pad

        rect = plt.Rectangle((left, bottom), side, side, edgecolor="white", facecolor='none', linewidth=0.5)
        ax.add_patch(rect)

        ax.arrow(x_spaxel_center, y_spaxel_center, dx, dy,
                head_width=5, head_length=2, fc=color, ec=color, alpha=1, width=1)



    ax = axs[1]
    ax.imshow(hst_rebinned.data, origin='lower', cmap='gray', extent=extent_highres_on_lowres)
    for x_spaxel_center, y_spaxel_center, dx, dy, length, x_bin, y_bin in arrows:
        color = cmap(norm(length*hst_pixscale))

        pad = 0.01 * scale_highres_per_lowres
        left = extent_lowres_on_hst[0] + x_bin * scale_highres_per_lowres + pad
        bottom = extent_lowres_on_hst[2] + y_bin * scale_highres_per_lowres + pad
        side = scale_highres_per_lowres - 2 * pad

        rect = plt.Rectangle((left, bottom), side, side, edgecolor="white", facecolor='none', linewidth=0.5)
        ax.add_patch(rect)

        ax.arrow(x_spaxel_center, y_spaxel_center, dx, dy,
                head_width=5, head_length=2, fc=color, ec=color, alpha=1, width=1)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([1.0, 0.05, 0.03, 0.85])
    cbar = plt.colorbar(sm, cax=cax, label='Offset length [arcsec]')
    cbar.ax.tick_params(labelsize=12, color='k')
    cbar.set_label('Offset length [arcsec]', fontsize=14, color='k')



    plt.tight_layout()
    fig.savefig(out_filename, dpi=300)