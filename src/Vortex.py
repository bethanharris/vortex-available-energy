import numpy as np


class Vortex:
    f = 6.14e-5
    g = 9.81
    Rd = 287.
    p0 = 100000.
    cp = 1004.5
    far_field_pressure = 100000.
    radial_coefficient = 1.048
    radial_edge = 200000.
    vertical_top = 16000.

    def __init__(self, t_surf, gamma, p_centre, rs, zs):
        self.surface_temperature = t_surf
        self.lapse_rate = gamma
        self.central_pressure = p_centre
        self.pressure_deficit = self.central_pressure - self.far_field_pressure
        self.radial_scale = rs
        self.vertical_scale = zs

    def grid(self, return_lists=False):
        r_list = np.linspace(1., self.radial_edge, int(self.radial_edge / 1000. + 1.), endpoint=True)
        z_list = np.linspace(1., self.vertical_top, int(self.vertical_top / 1000. + 1.), endpoint=True)
        r_grid, z_grid = np.meshgrid(r_list, z_list)
        if return_lists:
            return r_list, z_list
        else:
            return r_grid, z_grid

    def gridded_variable(self, variable):
        r_grid, z_grid = self.grid()
        return variable(r_grid, z_grid)

    def pressure_perturbation(self, r, z):
        radial_term = 1. - np.exp(-self.radial_coefficient * self.radial_scale / r)
        vertical_term = np.exp(-z / self.vertical_scale) * np.cos(0.5 * np.pi * z / self.vertical_top)
        return self.pressure_deficit * radial_term * vertical_term

    def environment_temperature(self, z):
        return self.surface_temperature * (1. - self.lapse_rate * z)

    def environment_pressure(self, z):
        return self.far_field_pressure * np.exp(
            (self.g * np.log(1. - self.lapse_rate * z)) / (self.Rd * self.surface_temperature * self.lapse_rate))

    def environment_pressure_vertical_gradient(self, z):
        return -(self.far_field_pressure * self.g * (1. - self.lapse_rate * z) ** (
                    self.g / (self.Rd * self.surface_temperature * self.lapse_rate) - 1.)) / (
                           self.Rd * self.surface_temperature)

    def pressure(self, r, z):
        return self.environment_pressure(z) + self.pressure_perturbation(r, z)

    def pressure_vertical_gradient(self, r, z):
        radial_term = 1. - np.exp(-self.radial_coefficient * self.radial_scale / r)
        vertical_term = -np.exp(-z / self.vertical_scale) * (1. / self.vertical_scale) * (
                (0.5 * np.pi * self.vertical_scale / self.vertical_top) * np.sin(
            0.5 * np.pi * z / self.vertical_top) + np.cos(
            0.5 * np.pi * z / self.vertical_top))
        return self.pressure_deficit * radial_term * vertical_term + self.environment_pressure_vertical_gradient(z)

    def pressure_radial_gradient(self, r, z):
        radial_term = -self.radial_coefficient * self.radial_scale * np.exp(
            -self.radial_coefficient * self.radial_scale / r) / (r ** 2)
        vertical_term = np.exp(-z / self.vertical_scale) * np.cos(0.5 * np.pi * z / self.vertical_top)
        return self.pressure_deficit * radial_term * vertical_term

    def density(self, r, z):
        return -self.pressure_vertical_gradient(r, z) / self.g

    def temperature(self, r, z):
        return self.pressure(r, z) / (self.Rd * self.density(r, z))

    def gradient_wind_term(self, r, z):
        # fv + v^2/r
        return self.pressure_radial_gradient(r, z) / self.density(r, z)

    def azimuthal_wind(self, r, z):
        gradient_wind = self.gradient_wind_term(r, z)
        v = 0.5 * r * (-self.f + np.sqrt(self.f ** 2 + 4 * gradient_wind / r))
        return v

    def angular_momentum(self, r, z):
        return np.sqrt(r ** 3 * (self.gradient_wind_term(r, z) + 0.25 * r * self.f ** 2))

    def potential_temperature(self, r, z):
        return self.temperature(r, z) * (self.p0 / self.pressure(r, z)) ** (self.Rd / self.cp)

    def entropy(self, r, z):
        return self.cp * np.log(self.potential_temperature(r, z))

    def geopotential(self, z):
        return self.g * z

    def mu(self, r, z):
        return self.angular_momentum(r, z)**2

    def angular_momentum_from_azimuthal_wind(self, v, r):
        return r * v + 0.5 * self.f * r**2

    def azimuthal_wind_from_angular_momentum(self, M, r):
        return M/r - 0.5 * self.f * r

    @staticmethod
    def chi(r):
        return 1./(2. * r**2)


def smith_vortex():
    return Vortex(303., 2.12e-5, 95000., 50000., 8000.)


def perturbed_vortex():
    return Vortex(305., 2.12e-5, 93000., 30000., 8000.)
