import matplotlib.pyplot as plt
import matplotlib.cm as cm
from utils import grid_variable
from available_energy import *


def plot_variable(variable):
    plt.figure()
    plt.contourf(grid_variable(variable), cmap=cm.YlGn)
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


if __name__ == '__main__':
    r0_list = [100., 1000., 2500., 5000., 10000., 25000., 40000., 50000., 100000., 150000., 200000.]
    z0_list = [10., 100., 250., 500., 1000., 2500., 4000., 5000., 10000., 15000.]
    for r0 in r0_list:
        for z0 in z0_list:
            plot_available_energy_perturbations(r0, z0, show=False)
