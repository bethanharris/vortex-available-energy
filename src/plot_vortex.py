import matplotlib.pyplot as plt
import matplotlib.cm as cm
from available_energy import *


def r_z_grid():
    r_list = np.linspace(1., 200000., 201, endpoint=True)
    z_list = np.linspace(1., 16000., 17, endpoint=True)
    r_grid, z_grid = np.meshgrid(r_list, z_list)
    return r_grid, z_grid


def grid_variable(function):
    r_grid, z_grid = r_z_grid()
    return function(r_grid, z_grid)


def plot_variable(function):
    plt.figure()
    plt.contourf(grid_variable(function), cmap=cm.YlGn)
    plt.colorbar()
    plt.show()


def plot_available_energy_perturbations(r, z, show=True):
    base_M = angular_momentum(r, z)
    base_entropy = entropy(r, z)

    r_grid, z_grid = r_z_grid()
    all_M = angular_momentum(r_grid, z_grid)
    all_entropy = entropy(r_grid, z_grid)
    M_grid, entropy_grid = np.meshgrid(np.linspace(all_M.min(), all_M.max(), 100), np.linspace(all_entropy.min(), all_entropy.max(), 100))
    ae_M_eta = available_energy(M_grid, entropy_grid, r, z)
    ae_r_z = available_energy(angular_momentum(r_grid, z_grid), entropy(r_grid, z_grid), r, z)

    plt.figure()
    plt.contourf((M_grid-base_M)/1.e6, entropy_grid-base_entropy, ae_M_eta, 20, cmap=cm.gist_heat_r)
    plt.xlabel(r'$\mathregular{M - M_0\;\left(10^6\,m^2s^{-1}\right)}$', fontsize=18)
    plt.ylabel(r'$\mathregular{\eta - \eta_0\;\left(Jkg^{-1}K^{-1}\right)}$', fontsize=18)
    plt.gca().tick_params(labelsize=14)
    cbar = plt.colorbar()
    cbar.set_label(r'$\mathregular{\Pi\;\left(Jkg^{-1}\right)}$', fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    plt.title(r'$\mathregular{r_0 = %d \,km,\; z_0 = %d\, km}$' % (r/1000., z/1000.), fontsize=16)
    plt.tight_layout()
    plt.savefig('../results/ae_perturbation_M_eta_r_%d_z_%d.png' % (r, z), dpi=300)

    plt.figure()
    plt.contourf((r_grid-r)/1000., (z_grid-z)/1000., ae_r_z, 20, cmap=cm.gist_heat_r)
    plt.xlabel(r'$\mathregular{r - r_0\;\left(km\right)}$', fontsize=18)
    plt.ylabel(r'$\mathregular{z - z_0\;\left(km\right)}$', fontsize=18)
    plt.gca().tick_params(labelsize=14)
    cbar = plt.colorbar()
    cbar.set_label(r'$\mathregular{\Pi\;\left(Jkg^{-1}\right)}$', fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    plt.title(r'$\mathregular{r_0 = %d \,km,\; z_0 = %d\, km}$' % (r / 1000., z / 1000.), fontsize=16)
    plt.tight_layout()
    plt.savefig('../results/ae_perturbation_rz_r_%d_z_%d.png' % (r, z), dpi=300)

    if show:
        plt.show()
    else:
        plt.close()

    return M_grid-base_M, entropy_grid-base_entropy, ae


if __name__=='__main__':
    perturbation_M, perturbation_entropy, ae = plot_available_energy_perturbations(50000., 5000.)
