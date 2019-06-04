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


def plot_available_energy(r_ref, z_ref):
    r_list = np.linspace(r_ref - 10000., r_ref + 10000., 501, endpoint=True)
    z_list = np.linspace(z_ref - 1000., z_ref + 1000., 501, endpoint=True)
    r_grid, z_grid = np.meshgrid(r_list, z_list)
    M_ref = angular_momentum(r_ref, z_ref)
    M_grid = angular_momentum(r_grid, z_grid)
    entropy_ref = entropy(r_ref, z_ref)
    entropy_grid = entropy(r_grid, z_grid)
    ae = available_energy(r_grid, z_grid, r_ref, z_ref)
    plt.figure()
    plt.contourf(r_grid-r_ref, z_grid-z_ref, ae, 100, cmap=cm.seismic)
    plt.colorbar()
    plt.figure()
    plt.contourf(entropy_grid - entropy_ref, M_grid - M_ref, ae, 100, cmap=cm.seismic)
    plt.colorbar()
    plt.show()


def plot_available_energy_perturbations(r, z):
    base_M = angular_momentum(r, z)
    base_entropy = entropy(r, z)

    r_grid, z_grid = r_z_grid()
    all_M = angular_momentum(r_grid, z_grid)
    all_entropy = entropy(r_grid, z_grid)
    M_grid, entropy_grid = np.meshgrid(np.linspace(all_M.min(), all_M.max(), 100.), np.linspace(all_entropy.min(), all_entropy.max(), 100.))
    ae = available_energy(M_grid, entropy_grid, r, z)

    plt.figure()
    plt.contourf(M_grid-base_M, entropy_grid-base_entropy, ae, 20, cmap=cm.gist_heat_r)
    plt.colorbar()
    plt.show()
    return M_grid-base_M, entropy_grid-base_entropy, ae


if __name__=='__main__':
    perturbation_M, perturbation_entropy, ae = plot_available_energy_perturbations(50000., 5000.)
