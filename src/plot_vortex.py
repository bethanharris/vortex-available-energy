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
    plt.xlabel(r'$\mathregular{M - M_0\;\left(10^6\,m^2s^{-1}\right)}$', fontsize=18)
    plt.ylabel(r'$\mathregular{\eta - \eta_0\;\left(Jkg^{-1}K^{-1}\right)}$', fontsize=18)
    plt.gca().tick_params(labelsize=14)
    cbar = plt.colorbar()
    cbar.set_label(r'$\mathregular{A_e\;\left(Jkg^{-1}\right)}$', fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    plt.title(r'$\mathregular{r = %g \,km,\; z = %g\, km}$' % (r/1000., z/1000.), fontsize=16)
    plt.tight_layout()
    if save:
        plt.savefig('../results/ae_perturbation_M_eta_r_%d_z_%d.png' % (r, z), dpi=300)

    plt.figure()
    plt.contourf(perturbation_mu/1.e6, perturbation_p, ae_mu_p, 20, cmap=cm.gist_heat_r)
    plt.xlabel(r'$\mathregular{\mu - \mu_m\;\left(10^6\,m^4s^{-2}\right)}$', fontsize=18)
    plt.ylabel(r'$\mathregular{p_m - p_\star\;\left(Jkg^{-1}K^{-1}\right)}$', fontsize=18)
    plt.gca().tick_params(labelsize=14)
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
    plt.plot(perturbation_v, quadratic_perturbation_v, '--', color='gray', linewidth=2, label='$\left(v-v_m\\right)^2$')
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


def illustrate_lifting(vortex):
    r = 50000.
    z = 10000.
    M = vortex.angular_momentum(30000., 3000.)
    eta = vortex.entropy(30000., 3000.)
    r_grid, z_grid = vortex.grid()
    r_mu, z_mu = position_at_isobaric_surface(vortex, M, vortex.pressure(r, z))
    r_ref, z_ref = reference_position(vortex, M, eta)
    plt.contour(r_grid / 1000., z_grid / 1000., vortex.gridded_variable(vortex.pressure), colors='k', linestyles='--')
    plt.contour(r_grid / 1000., z_grid / 1000., vortex.gridded_variable(vortex.entropy), colors='k', linestyles='-')
    plt.contour(r_grid / 1000., z_grid / 1000., vortex.gridded_variable(vortex.angular_momentum), colors='k',
                linestyles=':')
    plt.plot(r/1000., z/1000., 'ro')
    plt.annotate('$\left(r, z\\right)$', xy=(r/1000., z/1000.), xytext=(5, 5), textcoords='offset points')
    plt.plot(r_mu/1000., z_mu/1000., 'go')
    plt.annotate('$\left(r_{\mu}, z_{\mu}\\right)$', xy=(r_mu / 1000., z_mu / 1000.), xytext=(-40, 5), textcoords='offset points')
    plt.plot(r_ref/1000., z_ref/1000., 'bo')
    plt.annotate('$\left(r_{\star}, z_{\star}\\right)$', xy=(r_ref / 1000., z_ref / 1000.), xytext=(-10, -15), textcoords='offset points')
    plt.tight_layout()


if __name__ == '__main__':
    vortex = smith_vortex()
    plot_available_energy_perturbations(vortex, 50000., 5000., show=False)
    illustrate_lifting(vortex)
