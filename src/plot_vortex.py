import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from derived_variables import *
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
    M_list = np.linspace(base_M*0.9, base_M*1.1, 100)
    entropy_list = np.linspace(base_entropy*0.9, base_entropy*1.1, 100)
    perturbation_M, perturbation_entropy = np.meshgrid(M_list, entropy_list)
    ae = available_energy(perturbation_M, perturbation_entropy, r, z)
    plt.figure()
    plt.contourf(perturbation_M-base_M, perturbation_entropy-base_entropy, ae, 100, cmap=cm.YlGn)
    plt.colorbar()
    plt.show()
    return perturbation_M, perturbation_entropy, ae


perturbation_M, perturbation_entropy, ae = plot_available_energy_perturbations(5000., 500.)
