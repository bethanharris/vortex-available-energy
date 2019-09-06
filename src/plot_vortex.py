import matplotlib.pyplot as plt
import matplotlib.cm as cm
from available_energy import *
from Vortex import *


def plot_variable(vortex, variable, label=''):
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


def plot_available_energy_perturbations(vortex, r, z, save=False, show=True):
    perturbation_M, perturbation_eta, ae_M_entropy = available_potential_energy_perturbations_M_entropy(vortex, r, z)
    perturbation_mu, perturbation_p, ae_mu_p = available_potential_energy_perturbations_mu_pressure(vortex, r, z)
    perturbation_r, perturbation_z, ae_r_z = available_potential_energy_perturbations_r_z(vortex, r, z)

    plt.figure()
    plt.contourf(perturbation_M/1.e6, perturbation_eta, ae_M_entropy, 20, cmap=cm.gist_heat_r)
    plt.xlabel(r'$\mathregular{M - M_m\;\left(10^6\,m^2s^{-1}\right)}$', fontsize=18)
    plt.ylabel(r'$\mathregular{\eta - \eta_m\;\left(Jkg^{-1}K^{-1}\right)}$', fontsize=18)
    plt.gca().tick_params(labelsize=14)
    plt.gca().set_facecolor('gray')
    cbar = plt.colorbar()
    cbar.set_label(r'$\mathregular{A_e\;\left(Jkg^{-1}\right)}$', fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    plt.title(r'$\mathregular{r = %g \,km,\; z = %g\, km}$' % (r/1000., z/1000.), fontsize=16)
    plt.tight_layout()
    if save:
        plt.savefig('../results/ae_perturbation_M_eta_r_%d_z_%d.png' % (r, z), dpi=300)

    plt.figure()
    plt.contourf(perturbation_mu/1.e13, perturbation_p/100., ae_mu_p, 20, cmap=cm.gist_heat_r)
    plt.xlabel(r'$\mathregular{\mu_m - \mu\;\left(10^{13}\,m^4s^{-2}\right)}$', fontsize=18)
    plt.ylabel(r'$\mathregular{p_m - p_\star\;\left(hPa\right)}$', fontsize=18)
    plt.gca().tick_params(labelsize=14)
    plt.gca().set_facecolor('gray')
    plt.gca().axhline(0, color='k', alpha=0.5)
    plt.gca().axvline(0, color='k', alpha=0.5)
    cbar = plt.colorbar()
    cbar.set_label(r'$\mathregular{A_e\;\left(Jkg^{-1}\right)}$', fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    plt.title(r'$\mathregular{r = %g \,km,\; z = %g\, km}$' % (r/1000., z/1000.), fontsize=16)
    plt.tight_layout()
    if save:
        plt.savefig('../results/ae_perturbation_mu_p_r_%d_z_%d.png' % (r, z), dpi=300)

    plt.figure()
    plt.contourf(perturbation_r/1000., perturbation_z/1000., ae_r_z, 20, cmap=cm.gist_heat_r)
    plt.xlabel(r'$\mathregular{r_\star - r\;\left(km\right)}$', fontsize=18)
    plt.ylabel(r'$\mathregular{z_\star - z\;\left(km\right)}$', fontsize=18)
    plt.gca().tick_params(labelsize=14)
    plt.gca().set_facecolor('gray')
    cbar = plt.colorbar()
    cbar.set_label(r'$\mathregular{A_e\;\left(Jkg^{-1}\right)}$', fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    plt.title(r'$\mathregular{r = %g \,km,\; z = %g\, km}$' % (r / 1000., z / 1000.), fontsize=16)
    plt.tight_layout()
    if save:
        plt.savefig('../results/ae_perturbation_rzref_r_%d_z_%d.png' % (r, z), dpi=300)

    if show:
        plt.show()
    else:
        plt.close('all')

    return


def plot_kinetic_energy_perturbations(vortex, r, z, v_range, save=False, show=True):
    perturbation_v, perturbation_pi_k, quadratic_perturbation_v = pi_k_perturbations(vortex, r, z, v_range)

    plt.figure()
    plt.plot(perturbation_v, perturbation_pi_k, 'k-', linewidth=2, label='$\Pi_k$')
    plt.plot(perturbation_v, quadratic_perturbation_v, '--', color='gray', linewidth=2, label='$\\frac{\left(v-v_m\\right)^2}{2}$')
    plt.xlabel(r'$\mathregular{v - v_m\;\left(ms^{-1}\right)}$', fontsize=18)
    plt.ylabel(r'$\mathregular{\left(Jkg^{-1}\right)}$', fontsize=18)
    plt.gca().tick_params(labelsize=14)
    plt.legend(fontsize=18)
    plt.title(r'$\mathregular{r = %g \,km,\; z = %g\, km,\; v_m = %0.1f ms^{-1}}$' % (
    r / 1000., z / 1000., vortex.azimuthal_wind(r, z)), fontsize=16)
    plt.tight_layout()
    if save:
        plt.savefig('../results/ke_perturbation_r_%d_z_%d.png' % (r, z), dpi=300)

    if show:
        plt.show()
    else:
        plt.close('all')

    return


def plot_kinetic_energy_perturbation_ratio(vortex, r, z, v_range, save=False, show=True):
    perturbation_v, perturbation_pi_k, quadratic_perturbation_v = pi_k_perturbations(vortex, r, z, v_range)
    ratio = quadratic_perturbation_v/perturbation_pi_k
    zeros = np.where(perturbation_v == 0.)
    ratio[zeros] = 1.
    plt.figure()
    plt.plot(perturbation_v, ratio, 'k-', linewidth=2)
    plt.xlabel(r'$\mathregular{v - v_m\;\left(ms^{-1}\right)}$', fontsize=18)
    plt.ylabel(r'$\mathregular{\frac{\left(v - v_m\right)^2}{2\Pi_k}}$', fontsize=18)
    plt.gca().tick_params(labelsize=14)
    plt.title(r'$\mathregular{r = %g \,km,\; z = %g\, km,\; v_m = %0.1f ms^{-1}}$' % (
    r / 1000., z / 1000., vortex.azimuthal_wind(r, z)), fontsize=16)
    plt.tight_layout()
    if save:
        plt.savefig('../results/ke_perturbation_ratio_r_%d_z_%d.png' % (r, z), dpi=300)
    if show:
        plt.show()
    else:
        plt.close('all')
    return


def illustrate_lifting(vortex):
    r = 50000.
    z = 10000.
    M = vortex.angular_momentum(30000., 3000.)
    eta = vortex.entropy(30000., 3000.)
    r_grid, z_grid = vortex.grid()
    r_mu, z_mu = position_at_isobaric_surface(vortex, M, vortex.pressure(r, z))
    r_ref, z_ref = reference_position(vortex, M, eta)
    r_isobar_grid, z_isobar_grid = np.meshgrid(np.linspace(r_mu, r, 100, endpoint=True), np.linspace(z_mu, z, 100, endpoint=True))
    r_M_grid, z_M_grid = np.meshgrid(np.linspace(r_ref, r_mu, 100, endpoint=True), np.linspace(z_ref, z_mu, 100, endpoint=True))
    plt.contour(r_grid / 1000., z_grid / 1000., vortex.gridded_variable(vortex.pressure), colors='k', linestyles='--')
    plt.contour(r_grid / 1000., z_grid / 1000., vortex.gridded_variable(vortex.entropy), colors='k', linestyles='-')
    plt.contour(r_grid / 1000., z_grid / 1000., vortex.gridded_variable(vortex.angular_momentum), colors='k',
                linestyles=':')
    plt.contour(vortex.pressure(r_isobar_grid, z_isobar_grid), colors='orange', levels=[vortex.pressure(r, z)], extent=(r/1000.,r_mu/1000., z/1000., z_mu/1000.))
    plt.contour(vortex.angular_momentum(r_M_grid, z_M_grid), colors='orange', levels=[vortex.angular_momentum(r_ref, z_ref)], extent=(r_ref/1000., r_mu/1000., z_ref/1000., z_mu/1000.))
    plt.xlabel('r (km)', fontsize=18)
    plt.ylabel('z (km)', fontsize=18)
    plt.gca().tick_params(labelsize=14)
    plt.plot(r/1000., z/1000., 'ko')
    plt.annotate('$\left(r, z\\right)$', xy=(r/1000., z/1000.), xytext=(10, 5), textcoords='offset points', bbox=dict(boxstyle="round", fc="w"))
    plt.plot(r_mu/1000., z_mu/1000., 'ko')
    plt.annotate('$\left(r_{\mu}, z_{\mu}\\right)$', xy=(r_mu / 1000., z_mu / 1000.), xytext=(-42, 5), textcoords='offset points', bbox=dict(boxstyle="round", fc="w"))
    plt.plot(r_ref/1000., z_ref/1000., 'ko')
    plt.annotate('$\left(r_{\star}, z_{\star}\\right)$', xy=(r_ref / 1000., z_ref / 1000.), xytext=(-10, -20), textcoords='offset points', bbox=dict(boxstyle="round", fc="w"))
    plt.tight_layout()
    plt.savefig('../results/lifting_illustration.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    vortex = smith_vortex()
    plot_available_energy_perturbations(vortex, 50000., 5000.)
    illustrate_lifting(vortex)
