import numpy as np
import matplotlib.pyplot as plt


f = 5e-5


def ratio(r_max, v_max):
    num = 2 * (r_max ** 4) * (v_max ** 2) + 3 * (r_max ** 5) * f * v_max + f ** 2 * (r_max ** 6)
    denom = 4 * (v_max ** 2) * (r_max ** 4) + f ** 2 * (r_max ** 6) + 4 * v_max * f * (r_max ** 5)
    return num / denom


def plot_ratio(save=False):
    all_r = np.logspace(1, 6, num=1000)
    all_v = np.linspace(1., 100., 100., endpoint=True)
    R, V = np.meshgrid(all_r, all_v)
    ratios = ratio(R, V)
    plt.figure(figsize=(6, 4.5))
    ax = plt.gca()
    levels = np.linspace(0.5, 1., 11, endpoint=True)
    plt.contourf(R/1000., V, ratios, levels=levels, vmin=0.5, vmax=1.)
    ax.set_xlabel(r'$\mathregular{r_{max}\; \left(km\right)}$', fontsize=20)
    ax.set_ylabel(r'$\mathregular{v_{max}\; \left(ms^{-1}\right)}$', fontsize=20)
    ax.tick_params(labelsize=16)
    plt.title(r'$\mathregular{f=5\times 10^{-5}\;rad\,s^{-1}}$', fontsize=12)
    cbar = plt.colorbar()
    cbar.set_label(r'$\mathregular{\frac{\left(v - v_m\right)^2}{2\Pi_k}}$', fontsize=24, labelpad=5)
    cbar.ax.tick_params(labelsize=16)
    plt.tight_layout()
    if save:
        plt.savefig('../results/vmax_pi_k_ratio.png', dpi=400)
    plt.show()


if __name__ == '__main__':
    plot_ratio(save=True)
