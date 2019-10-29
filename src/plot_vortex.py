import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter
from Vortex import Vortex
from available_energy import (reference_position, position_at_isobaric_surface, pi_k_perturbations,
                              available_potential_energy_perturbations_M_entropy,
                              available_potential_energy_perturbations_mu_pressure,
                              available_potential_energy_perturbations_r_z)


def format_sci_string(x, decimal_places):
    """String formatter for standard form numbers.

    Parameters
    ----------
    x: Number to format
    decimal_places: Number of decimal places to include

    Returns
    -------
    String formatted as 'a \times 10^b'
    """
    s = '{x:0.{n:d}e}'.format(x=x, n=decimal_places)
    mantissa, exponent = s.split('e')
    return r'{m:s}\times 10^{{{e:d}}}'.format(m=mantissa, e=int(exponent))


def plot_variable(vortex, variable, label=''):
    """Generic filled-contour plotting function for gridded fields of vortex.

    Parameters
    ----------
    vortex: instance of Vortex class for vortex to be plotted
    variable: method of Vortex class e.g. vortex.azimuthal_wind
    label: (kwarg) string label for colorbar
    """
    plt.figure()
    r_grid, z_grid = vortex.grid()
    plt.contourf(r_grid/1000., z_grid/1000., vortex.gridded_variable(variable), cmap=cm.YlGn)
    plt.xlabel(r'$\mathregular{r\;\left(km\right)}$', fontsize=18)
    plt.ylabel(r'$\mathregular{z\;\left(km\right)}$', fontsize=18)
    plt.gca().tick_params(labelsize=14)
    cbar = plt.colorbar()
    cbar.set_label(label, fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    plt.tight_layout()
    plt.show()


def plot_azimuthal_wind(vortex):
    """Plot contours of vortex azimuthal wind.

    Parameters
    ----------
    vortex: instance of Vortex class
    """
    v = vortex.azimuthal_wind
    grid_v = vortex.gridded_variable(v)
    r_grid, z_grid = vortex.grid()
    plt.figure(figsize=(6, 4.5))
    ax = plt.gca()
    v_ctr = ax.contour(r_grid/1000., z_grid/1000., grid_v, colors='k')
    ax.clabel(v_ctr, v_ctr.levels, inline=True, fmt='%d', fontsize=14, colors='k')
    ax.set_xlabel(r'$\mathregular{r\;\left(km\right)}$', fontsize=22)
    ax.set_ylabel(r'$\mathregular{z\;\left(km\right)}$', fontsize=22)
    ax.tick_params(labelsize=18)
    ax.yaxis.set_ticks(np.arange(0, 17, 4))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    plt.tight_layout()
    plt.savefig('../results/v_contour.pdf')
    plt.show()


def plot_available_energy_perturbations(vortex, r, z, title=True, save=False, show=True):
    """Plot contours of vortex available energy for perturbations in M/entropy, mu/p_* and r/z at a fixed point.

    Parameters
    ----------
    vortex: instance of Vortex class determining reference state
    r: fixed radius (m)
    z: fixed height (m)
    title: (kwarg, bool) show fixed r/z position as figure title?
    save: (kwarg, bool, default=False) save plots?
    show: (kwarg, bool, default=True) show plots interactively?
    """
    perturbation_M, perturbation_eta, ae_M_entropy = available_potential_energy_perturbations_M_entropy(vortex, r, z)
    perturbation_mu, perturbation_p, ae_mu_p = available_potential_energy_perturbations_mu_pressure(vortex, r, z)
    perturbation_r, perturbation_z, ae_r_z = available_potential_energy_perturbations_r_z(vortex, r, z)

    plt.figure(figsize=(6, 4.5))
    ax = plt.gca()
    plt.contourf(perturbation_M/1.e6, perturbation_eta, ae_M_entropy, 20, cmap=cm.gist_heat_r)
    plt.xlabel(r'$\mathregular{M - M_m\;\left(10^6\,m^2s^{-1}\right)}$', fontsize=20)
    plt.ylabel(r'$\mathregular{\eta - \eta_m\;\left(Jkg^{-1}K^{-1}\right)}$', fontsize=20)
    ax.tick_params(labelsize=16)
    ax.set_facecolor('gray')
    fmt = mpl.ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    cbar = plt.colorbar(format=fmt)
    cbar.set_label(r'$\mathregular{A_e\;\left(Jkg^{-1}\right)}$', fontsize=18, labelpad=5)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.yaxis.offsetText.set(size=16)
    if title:
        plt.title(r'$\mathregular{r = %g \,km,\; z = %g\, km}$' % (r/1000., z/1000.), fontsize=14)
    plt.tight_layout()
    if save:
        plt.savefig('../results/ae_perturbation_M_eta_r_%d_z_%d.png' % (r, z), dpi=400)
        plt.savefig('../results/ae_perturbation_M_eta_r_%d_z_%d.pdf' % (r, z))

    plt.figure(figsize=(6, 4.5))
    ax = plt.gca()
    plt.contourf(perturbation_mu/1.e13, perturbation_p/100., ae_mu_p, 20, cmap=cm.gist_heat_r)
    plt.xlabel(r'$\mathregular{\mu_m - \mu\;\left(10^{13}\,m^4s^{-2}\right)}$', fontsize=20)
    plt.ylabel(r'$\mathregular{p_m - p_\star\;\left(hPa\right)}$', fontsize=20)
    ax.tick_params(labelsize=16)
    ax.set_facecolor('gray')
    ax.axhline(0, color='k', alpha=0.5)
    ax.axvline(0, color='k', alpha=0.5)
    cbar = plt.colorbar(format=fmt)
    cbar.set_label(r'$\mathregular{A_e\;\left(Jkg^{-1}\right)}$', fontsize=18, labelpad=5)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.yaxis.offsetText.set(size=16)
    if title:
        plt.title(r'$\mathregular{r = %g \,km,\; z = %g\, km}$' % (r/1000., z/1000.), fontsize=16)
    plt.tight_layout()
    if save:
        plt.savefig('../results/ae_perturbation_mu_p_r_%d_z_%d.png' % (r, z), dpi=400)
        plt.savefig('../results/ae_perturbation_mu_p_r_%d_z_%d.pdf' % (r, z))

    plt.figure(figsize=(6, 4.5))
    ax = plt.gca()
    plt.contourf(perturbation_r/1000., perturbation_z/1000., ae_r_z, 20, cmap=cm.gist_heat_r)
    plt.xlabel(r'$\mathregular{r_\star - r\;\left(km\right)}$', fontsize=20)
    plt.ylabel(r'$\mathregular{z_\star - z\;\left(km\right)}$', fontsize=20)
    ax.tick_params(labelsize=16)
    ax.set_facecolor('gray')
    cbar = plt.colorbar(format=fmt)
    cbar.set_label(r'$\mathregular{A_e\;\left(Jkg^{-1}\right)}$', fontsize=18, labelpad=5)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.yaxis.offsetText.set(size=16)
    if title:
        plt.title(r'$\mathregular{r = %g \,km,\; z = %g\, km}$' % (r / 1000., z / 1000.), fontsize=14)
    plt.tight_layout()
    if save:
        plt.savefig('../results/ae_perturbation_rzref_r_%d_z_%d.png' % (r, z), dpi=400)
        plt.savefig('../results/ae_perturbation_rzref_r_%d_z_%d.pdf' % (r, z))

    if show:
        plt.show()
    else:
        plt.close('all')

    return


def plot_kinetic_energy_perturbations(vortex, r, z, v_range, save=False, show=True):
    """Plot comparison of eddy kinetic energy and Pi_k for perturbations around reference azimuthal wind at a point.

    Parameters
    ----------
    vortex: instance of Vortex class determining reference state
    r: radius (m)
    z: height (m)
    v_range: maximum range in azimuthal wind to sample around reference wind (in both directions) (m/s)
    save: (kwarg, bool, default=False) save plots?
    show: (kwarg, bool, default=True) show plots interactively?
    """
    perturbation_v, perturbation_pi_k, quadratic_perturbation_v = pi_k_perturbations(vortex, r, z, v_range)
    plt.figure(figsize=(6, 4.5))
    ax = plt.gca()
    plt.plot(perturbation_v, perturbation_pi_k, 'k-', linewidth=2, label='$\Pi_k$')
    plt.plot(perturbation_v, quadratic_perturbation_v, '--', color='gray', linewidth=2,
                 label='$\\frac{\left(v-v_m\\right)^2}{2}$')
    plt.xlabel(r'$\mathregular{v - v_m\;\left(ms^{-1}\right)}$', fontsize=20)
    plt.ylabel(r'$\mathregular{\left(Jkg^{-1}\right)}$', fontsize=20)
    ax.tick_params(labelsize=16)
    plt.legend(fontsize=20)
    fmt = mpl.ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.offsetText.set(size=16)
    base_v = vortex.azimuthal_wind(r, z)
    if base_v < 0.1:
        plt.title(r'$\mathregular{v_m = %s \;ms^{-1}}$' % (format_sci_string(base_v, 0)), fontsize=16)
    else:
        plt.title(r'$\mathregular{v_m = %0.1f \;ms^{-1}}$' % base_v, fontsize=16)
    plt.tight_layout()
    save_name = '../results/ke_perturbation_r_%d_z_%d' % (r, z)
    if save:
        plt.savefig(save_name + '.png', dpi=400)
        plt.savefig(save_name + '.pdf')
    if show:
        plt.show()
    else:
        plt.close('all')
    return


def plot_jacobian_ratio(vortex, save=False, show=True):
    """Plot ratio of vortex Jacobian to resting Jacobian over domain.

    Parameters
    ----------
    vortex: instance of Vortex class determining reference state
    save: (kwarg, bool, default=False) save plot?
    show: (kwarg, bool, default=True) show plot interactively?
    """
    plt.figure(figsize=(6, 4.5))
    ax = plt.gca()
    r_grid, z_grid = vortex.grid()
    plt.contourf(r_grid/1000., z_grid/1000., vortex.gridded_variable(vortex.jacobian_ratio), cmap=cm.Blues)
    ax.set_xlabel(r'$\mathregular{r\;\left(km\right)}$', fontsize=20)
    ax.set_ylabel(r'$\mathregular{z\;\left(km\right)}$', fontsize=20)
    ax.tick_params(labelsize=16)
    ax.yaxis.set_ticks(np.arange(0, 17, 4))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    cbar = plt.colorbar()
    cbar.set_label(r'$\mathregular{\frac{J_{\mu p}}{J_{\mu p_0}}}$', fontsize=24)
    cbar.ax.tick_params(labelsize=16)
    plt.tight_layout()
    if save:
        plt.savefig('../results/jacobian_ratio.png', dpi=400)
    if show:
        plt.show()
    else:
        plt.close()


def plot_jacobian_eke_ratio(vortex, zoom=False, save=False, show=True):
    """Plot ratio of eddy kinetic energy to vortex Jacobian over domain.

    Parameters
    ----------
    vortex: instance of Vortex class determining reference state
    zoom: (kwarg, bool, default=False) plot inner 50 km only?
    save: (kwarg, bool, default=False) save plot? #TODO: make save default for all plots
    show: (kwarg, bool, default=True) show plot interactively?
    """
    ratios = vortex.gridded_variable(vortex.eddy_kinetic_energy_ratio)
    plt.figure(figsize=(6, 4.5))
    ax = plt.gca()
    r_grid, z_grid = vortex.grid()
    levels = np.linspace(0., 3.5, 8, endpoint=True)
    plt.contourf(r_grid/1000., z_grid/1000., ratios, levels, cmap=cm.Purples)
    line_ctrs = ax.contour(r_grid/1000., z_grid/1000., ratios, [0.5, 1.], colors='k')
    ax.clabel(line_ctrs, line_ctrs.levels, inline=True, fmt='%g', fontsize=14, colors='k')
    ax.set_xlabel(r'$\mathregular{r\;\left(km\right)}$', fontsize=20)
    ax.set_ylabel(r'$\mathregular{z\;\left(km\right)}$', fontsize=20)
    ax.tick_params(labelsize=16)
    ax.yaxis.set_ticks(np.arange(0, 17, 4))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    cbar = plt.colorbar(ticks=levels)
    cbar.set_label(r'$\mathregular{\frac{\left(v - v_m\right)^2}{2\Pi_k}}$', fontsize=24)
    cbar.ax.tick_params(labelsize=16)
    save_name = '../results/jacobian_eke_pik_ratio'
    if zoom:
        ax.set_xlim([0., 50.])
        save_name += '_zoom'
    plt.tight_layout()
    if save:
        plt.savefig('{n:s}.png'.format(n=save_name), dpi=400)
    if show:
        plt.show()
    else:
        plt.close()


def illustrate_lifting(save=False):
    """Plot schematic illustrating the path along which a parcel is lifted to compute vortex available energy.

    Uses parcel at r = 30km, z = 3km in Smith (2005) vortex, with M and entropy set to give a reference position
    at r_* = 75km, z_* = 11km. Label positioning will need tweaking if example parcel is changed.

    Parameters
    ----------
    save: (kwarg, bool, default=False) save plot? #TODO: add show as kwarg
    """
    # Set example parcel properties
    vortex = Vortex.smith()
    r = 30000.
    z = 3000.
    M = vortex.angular_momentum(75000., 11000.)
    eta = vortex.entropy(75000., 11000.) #TODO: rename eta to entropy everywhere
    # Find reference position and point to break integral path into thermodynamic/mechanical for example parcel
    r_mu, z_mu = position_at_isobaric_surface(vortex, M, vortex.pressure(r, z))
    r_ref, z_ref = reference_position(vortex, M, eta)

    # Get complete r/z grid, and restricted grids that only cover branches of integral path
    r_grid, z_grid = vortex.grid()
    r_isobar_grid, z_isobar_grid = np.meshgrid(np.linspace(r_mu, r, 100, endpoint=True),
                                               np.linspace(z_mu, z, 100, endpoint=True))
    r_M_grid, z_M_grid = np.meshgrid(np.linspace(r_ref, r_mu, 100, endpoint=True),
                                     np.linspace(z_ref, z_mu, 100, endpoint=True))

    # Plot reference state isobars, isentropes and surfaces of constant angular momentum
    plt.figure(figsize=(6, 4.5))
    ax = plt.gca()
    plt.contour(r_grid / 1000., z_grid / 1000., vortex.gridded_variable(vortex.pressure), colors='k', linestyles='--')
    plt.contour(r_grid / 1000., z_grid / 1000., vortex.gridded_variable(vortex.entropy), colors='k', linestyles='-')
    plt.contour(r_grid / 1000., z_grid / 1000., vortex.gridded_variable(vortex.angular_momentum), colors='k',
                linestyles=':')
    # Plot contours to show integration paths
    plt.contour(vortex.pressure(r_isobar_grid, z_isobar_grid), colors='orange', levels=[vortex.pressure(r, z)],
                extent=(r / 1000., r_mu / 1000., z / 1000., z_mu / 1000.), linewidths=3)
    plt.contour(vortex.angular_momentum(r_M_grid, z_M_grid), colors='orange',
                levels=[vortex.angular_momentum(r_ref, z_ref)],
                extent=(r_ref / 1000., r_mu / 1000., z_ref / 1000., z_mu / 1000.), linewidths=3)
    # Plot & label points to show parcel's original position, reference position and position where path integral breaks
    plt.plot(r/1000., z/1000., 'ko')
    plt.annotate('$\left(r, z\\right)$', xy=(r / 1000., z / 1000.), xytext=(-18, -23), textcoords='offset points',
                 bbox=dict(boxstyle="round", fc="w"), fontsize=14)
    plt.plot(r_mu/1000., z_mu/1000., 'ko')
    plt.annotate('$\left(r_{\mu}, z_{\mu}\\right)$', xy=(r_mu / 1000., z_mu / 1000.), xytext=(13, -7),
                 textcoords='offset points', bbox=dict(boxstyle="round", fc="w"), fontsize=14)
    plt.plot(r_ref/1000., z_ref/1000., 'ko')
    plt.annotate('$\left(r_{\star}, z_{\star}\\right)$', xy=(r_ref / 1000., z_ref / 1000.), xytext=(15, 0),
                 textcoords='offset points', bbox=dict(boxstyle="round", fc="w"), fontsize=14)
    # General formatting of plot area
    plt.xlabel('r (km)', fontsize=20)
    plt.ylabel('z (km)', fontsize=20)
    ax.tick_params(labelsize=16)
    ax.yaxis.set_ticks(np.arange(0, 17, 4))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    plt.tight_layout()
    if save:
        plt.savefig('../results/lifting_illustration.pdf')
    plt.show()


if __name__ == '__main__':
    smith_vortex = Vortex.smith()
    plot_available_energy_perturbations(smith_vortex, 40000., 5000.)
    illustrate_lifting(smith_vortex)
