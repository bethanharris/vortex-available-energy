import numpy as np


def r_z_grid():
    r_list = np.linspace(1., 200000., 201, endpoint=True)
    z_list = np.linspace(1., 16000., 17, endpoint=True)
    r_grid, z_grid = np.meshgrid(r_list, z_list)
    return r_grid, z_grid


def grid_variable(variable):
    r_grid, z_grid = r_z_grid()
    return variable(r_grid, z_grid)