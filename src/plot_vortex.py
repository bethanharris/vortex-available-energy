import matplotlib.pyplot as plt
import matplotlib.cm as cm
from utils import grid_variable, r_z_grid
from available_energy import *


def plot_variable(function):
    plt.figure()
    plt.contourf(grid_variable(function), cmap=cm.YlGn)
    plt.colorbar()
    plt.show()


def plot_available_energy_perturbations(r, z, show=True):
    pertubation_M, perturbation_eta, ae_M_eta = available_energy_perturbations_M_eta(r, z)
    pertubation_r, perturbation_z, ae_r_z = available_energy_perturbations_r_z(r, z)

    plt.figure()
    plt.contourf((pertubation_M)/1.e6, perturbation_eta, ae_M_eta, 20, cmap=cm.gist_heat_r)
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
    plt.contourf((pertubation_r)/1000., (perturbation_z)/1000., ae_r_z, 20, cmap=cm.gist_heat_r)
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

    return


if __name__=='__main__':
    plot_available_energy_perturbations(50000., 5000.)
