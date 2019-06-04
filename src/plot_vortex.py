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


def plot_available_energy_perturbations(r, z):
    base_M = angular_momentum(r, z)
    base_entropy = entropy(r, z)

    r_grid, z_grid = r_z_grid()
    all_M = angular_momentum(r_grid, z_grid)
    all_entropy = entropy(r_grid, z_grid)
    M_grid, entropy_grid = np.meshgrid(np.linspace(all_M.min(), all_M.max(), 100), np.linspace(all_entropy.min(), all_entropy.max(), 100))
    ae = available_energy(M_grid, entropy_grid, r, z)

    plt.figure()
    plt.contourf(M_grid-base_M, entropy_grid-base_entropy, ae, 20, cmap=cm.gist_heat_r)
    plt.colorbar()
    plt.show()
    return M_grid-base_M, entropy_grid-base_entropy, ae


if __name__=='__main__':
    perturbation_M, perturbation_entropy, ae = plot_available_energy_perturbations(50000., 5000.)
