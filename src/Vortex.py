import numpy as np


class Vortex:
    """Dry axisymmetric atmospheric vortex.

    Attributes
    ----------
    g: gravitational acceleration = 9.81 m/s^2
    Rd: specific gas constant for dry air = 287 J/kg/K
    cp: specific heat capacity at constant pressure for dry air = 1004.5 J/kg/K
    p0: reference pressure for computing potential temperature = 10^5 Pa
    far_field_pressure: ambient surface pressure = 10^5 Pa
    radial_coefficient: radial scaling coefficient x = 1.048
    radial_edge: outer radial extent of default domain = 200 km
    vertical_top: upper vertical extent of default domain = 16 km
    """

    g = 9.81
    Rd = 287.
    cp = 1004.5
    p0 = 100000.
    far_field_pressure = 100000.
    radial_coefficient = 1.048
    radial_edge = 200000.
    vertical_top = 16000.

    def __init__(self, t_surf, gamma, p_centre, f, rs, zs):
        """Initialise vortex.

        Parameters
        ----------
        t_surf: Ambient surface temperature (K)
        gamma: Constant lapse rate (m^-1)
        p_centre: Surface pressure at vortex centre (Pa)
        f: Coriolis parameter (rad/s)
        rs: Approximate radius of maximum wind (m)
        zs: Scale of vertical decay for pressure perturbation (m)
        """
        self.surface_temperature = t_surf
        self.lapse_rate = gamma
        self.central_pressure = p_centre
        self.pressure_deficit = self.central_pressure - self.far_field_pressure
        self.f = f
        self.radial_scale = rs
        self.vertical_scale = zs

    def grid(self, return_lists=False):
        """Create radius-height grid from r=1km to radial_edge, z=0km to vertical top, 1km spacing in both dimensions.

        Parameters
        ----------
        return_lists: Determine whether to return 1D (True) or 2D (False, default) arrays of r/z coordinates.

        Returns
        -------
        If return_lists False: 2D numpy arrays r_grid, z_grid specifying grids of r/z coordinates.
        If return_lists True: 1D numpy arrays r_list, z_list specifying r/z points used on grid.
        """
        r_list = np.linspace(1000., self.radial_edge, int(self.radial_edge / 1000.), endpoint=True)
        z_list = np.linspace(0., self.vertical_top, int(self.vertical_top / 1000.) + 1, endpoint=True)
        r_grid, z_grid = np.meshgrid(r_list, z_list)
        if return_lists:
            return r_list, z_list
        else:
            return r_grid, z_grid

    def gridded_variable(self, variable):
        """Compute variable on grid.

        Parameters
        ----------
        variable: method of Vortex class specifying field to compute

        Returns
        -------
        2D numpy array of variable computed on regular radius-height grid.
        """
        r_grid, z_grid = self.grid()
        return variable(r_grid, z_grid)

    def pressure_perturbation(self, r, z):
        """Compute perturbation from environmental pressure profile at a point.

        Parameters
        ----------
        r: radius (m)
        z: height (m)
        Accepted as floats to compute for single point or numpy arrays to compute for many points.

        Returns
        -------
        Pressure perturbation (Pa)
        """
        radial_term = 1. - np.exp(-self.radial_coefficient * self.radial_scale / r)
        vertical_term = np.exp(-z / self.vertical_scale) * np.cos(0.5 * np.pi * z / self.vertical_top)
        return self.pressure_deficit * radial_term * vertical_term

    def environment_temperature(self, z):
        """Compute environmental temperature at specified height.

        Parameters
        ----------
        z: height (m)
        Accepted as float to compute for single height or numpy array to compute for many heights.

        Returns
        -------
        Environmental temperature (K)
        """
        return self.surface_temperature * (1. - self.lapse_rate * z)

    def environment_pressure(self, z):
        """Compute environmental pressure at specified height.

        Parameters
        ----------
        z: height (m)
        Accepted as float to compute for single height or numpy array to compute for many heights.

        Returns
        -------
        Environmental pressure (Pa)
        """
        return self.far_field_pressure * np.exp(
            (self.g * np.log(1. - self.lapse_rate * z)) / (self.Rd * self.surface_temperature * self.lapse_rate))

    def environment_pressure_vertical_gradient(self, z):
        """Compute vertical gradient of environmental pressure at specified height.

        Parameters
        ----------
        z: height (m)
        Accepted as float to compute for single height or numpy array to compute for many heights.

        Returns
        -------
        Vertical gradient of environmental pressure (Pa/m)
        """
        return -(self.far_field_pressure * self.g * (1. - self.lapse_rate * z) ** (
                    self.g / (self.Rd * self.surface_temperature * self.lapse_rate) - 1.)) / (
                           self.Rd * self.surface_temperature)

    def pressure(self, r, z):
        """Compute pressure at a point.

        Parameters
        ----------
        r: radius (m)
        z: height (m)
        Accepted as floats to compute for single point or numpy arrays to compute for many points.

        Returns
        -------
        Pressure (Pa)
        """
        return self.environment_pressure(z) + self.pressure_perturbation(r, z)

    def pressure_vertical_gradient(self, r, z):
        """Compute vertical gradient of pressure at a point.

        Parameters
        ----------
        r: radius (m)
        z: height (m)
        Accepted as floats to compute for single point or numpy arrays to compute for many points.

        Returns
        -------
        Vertical gradient of pressure (Pa/m)
        """
        radial_term = 1. - np.exp(-self.radial_coefficient * self.radial_scale / r)
        vertical_term = -np.exp(-z / self.vertical_scale) * (1. / self.vertical_scale) * (
                                (0.5 * np.pi * self.vertical_scale / self.vertical_top) * np.sin(
                                 0.5 * np.pi * z / self.vertical_top) + np.cos(
                                 0.5 * np.pi * z / self.vertical_top))
        return self.pressure_deficit * radial_term * vertical_term + self.environment_pressure_vertical_gradient(z)

    def pressure_radial_gradient(self, r, z):
        """Compute radial gradient of pressure at a point.

        Parameters
        ----------
        r: radius (m)
        z: height (m)
        Accepted as floats to compute for single point or numpy arrays to compute for many points.

        Returns
        -------
        Radial gradient of pressure (Pa/m)
        """
        radial_term = -self.radial_coefficient * self.radial_scale * np.exp(
            -self.radial_coefficient * self.radial_scale / r) / (r ** 2)
        vertical_term = np.exp(-z / self.vertical_scale) * np.cos(0.5 * np.pi * z / self.vertical_top)
        return self.pressure_deficit * radial_term * vertical_term

    def density(self, r, z):
        """Compute density at a point.

        Parameters
        ----------
        r: radius (m)
        z: height (m)
        Accepted as floats to compute for single point or numpy arrays to compute for many points.

        Returns
        -------
        Density (kg/m^3)
        """
        return -self.pressure_vertical_gradient(r, z) / self.g

    def temperature(self, r, z):
        """Compute temperature at a point.

        Parameters
        ----------
        r: radius (m)
        z: height (m)
        Accepted as floats to compute for single point or numpy arrays to compute for many points.

        Returns
        -------
        Temperature (K)
        """
        return self.pressure(r, z) / (self.Rd * self.density(r, z))

    def gradient_wind_term(self, r, z):
        """Compute gradient wind term fv + v^2/r at a point.

        Parameters
        ----------
        r: radius (m)
        z: height (m)
        Accepted as floats to compute for single point or numpy arrays to compute for many points.

        Returns
        -------
        Sum of Coriolis acceleration and centripetal acceleration in gradient wind balance (m/s^2)
        """
        return self.pressure_radial_gradient(r, z) / self.density(r, z)

    def azimuthal_wind(self, r, z):
        """Compute azimuthal wind (v) at a point.

        Parameters
        ----------
        r: radius (m)
        z: height (m)
        Accepted as floats to compute for single point or numpy arrays to compute for many points.

        Returns
        -------
        Azimuthal wind (m/s)
        """
        gradient_wind = self.gradient_wind_term(r, z)
        v = 0.5 * r * (-self.f + np.sqrt(self.f ** 2 + 4 * gradient_wind / r))
        return v

    def angular_momentum(self, r, z):
        """Compute angular momentum at a point.

        Parameters
        ----------
        r: radius (m)
        z: height (m)
        Accepted as floats to compute for single point or numpy arrays to compute for many points.

        Returns
        -------
        Angular momentum (m^2/s)
        """
        return np.sqrt(r ** 3 * (self.gradient_wind_term(r, z) + 0.25 * r * self.f ** 2))

    def potential_temperature(self, r, z):
        """Compute potential temperature at a point.

        Parameters
        ----------
        r: radius (m)
        z: height (m)
        Accepted as floats to compute for single point or numpy arrays to compute for many points.

        Returns
        -------
        Potential temperature (K)
        """
        return self.temperature(r, z) * (self.far_field_pressure / self.pressure(r, z)) ** (self.Rd / self.cp)

    def entropy(self, r, z):
        """Compute entropy at a point.

        Parameters
        ----------
        r: radius (m)
        z: height (m)
        Accepted as floats to compute for single point or numpy arrays to compute for many points.

        Returns
        -------
        Entropy (K)
        """
        return self.cp * np.log(self.potential_temperature(r, z))

    def geopotential(self, z):
        """Compute geopotential (g*z) at a height.

        Parameters
        ----------
        z: height (m)
        Accepted as float to compute for single height or numpy array to compute for many heights.

        Returns
        -------
        Geopotential (m^2/s^2)
        """
        return self.g * z

    def mu(self, r, z):
        """Compute mu (square of angular momentum) at a point.

        Parameters
        ----------
        r: radius (m)
        z: height (m)
        Accepted as floats to compute for single point or numpy arrays to compute for many points.

        Returns
        -------
        Square of angular momentum (m^4/s^2)
        """
        return self.angular_momentum(r, z)**2

    def angular_momentum_from_azimuthal_wind(self, v, r):
        """Compute angular momentum from azimuthal wind and radius.

        Parameters
        ----------
        v: azimuthal wind (m/s)
        r: radius (m)

        Returns
        -------
        Angular momentum (m^2/s)
        """
        return r * v + 0.5 * self.f * r**2

    def azimuthal_wind_from_angular_momentum(self, M, r):
        """Compute azimuthal wind from angular momentum and radius.

        Parameters
        ----------
        M: angular momentum (m^2/s)
        r: radius (m)

        Returns
        -------
        Azimuthal wind (m/s)
        """
        return M/r - 0.5 * self.f * r

    def dpsi_dr(self, r, z):
        """Find partial radial derivative of  psi = r^3/rho_m * (\partial p_m)/(\partial r) at a point.

        Parameters
        ----------
        r: radius (m)
        z: height (m)
        Accepted as floats to compute for single point or numpy arrays to compute for many points.

        Returns
        -------
        Value of (\partial psi)/(\partial r) (m^3/s^2)
        """
        a = -(self.central_pressure - self.far_field_pressure) * self.radial_coefficient * self.radial_scale * np.exp(
            -z / self.vertical_scale) * np.cos(0.5 * np.pi * z / self.vertical_top)
        b = self.radial_coefficient * self.radial_scale
        c = (self.central_pressure - self.far_field_pressure) / (self.g * self.vertical_scale) * np.exp(
            -z / self.vertical_scale) * ((0.5 * np.pi * self.vertical_scale / self.vertical_top) * np.sin(
            0.5 * np.pi * z / self.vertical_top) + np.cos(0.5 * np.pi * z / self.vertical_top))
        d = (self.far_field_pressure / (self.Rd * self.surface_temperature)) * (1. - self.lapse_rate * z) ** (
                    self.g / (self.Rd * self.surface_temperature * self.lapse_rate) - 1.)
        numerator = b * c * np.exp(b / r) + c * r * np.exp(b / r) + b * d * np.exp(b / r) + d * r * np.exp(
            b / r) - c * r
        denominator = r * (c * np.exp(b / r) + d * np.exp(b / r) - c) ** 2
        return a * numerator / denominator

    def dpsi_dz(self, r, z):
        """Find partial vertical derivative of  psi = r^3/rho_m * (\partial p_m)/(\partial z) at a point.

        Parameters
        ----------
        r: radius (m)
        z: height (m)
        Accepted as floats to compute for single point or numpy arrays to compute for many points.

        Returns
        -------
        Value of (\partial psi)/(\partial z) (m^3/s^2)
        """
        a = -(self.central_pressure - self.far_field_pressure) * np.exp(
            -(self.radial_coefficient * self.radial_scale) / r) * self.radial_coefficient * self.radial_scale * r
        b = (self.central_pressure - self.far_field_pressure) / (self.g * self.vertical_scale) * (
                    1. - np.exp(-(self.radial_coefficient * self.radial_scale) / r))
        c = self.far_field_pressure / (self.Rd * self.surface_temperature)
        d = self.g / (self.Rd * self.surface_temperature * self.lapse_rate) - 1.
        m = self.vertical_scale
        n = np.pi / (2. * self.vertical_top)
        p = self.lapse_rate
        numerator = m * n * (p * z - 1.) * np.sin(n * z) * (
                    b * m * n * np.sin(n * z) + c * np.exp(z / m) * (1. - p * z) ** d) + b * (m ** 2) * (n ** 2) * (
                                p * z - 1.) * np.cos(n * z) ** 2 + c * np.exp(z / m) * ((1. - p * z) ** d) * np.cos(
            n * z) * (d * m * p + p * z - 1.)
        denominator = m * (p * z - 1.) * (
                    b * m * n * np.sin(n * z) + b * np.cos(n * z) + c * np.exp(z / m) * (1. - p * z) ** d) ** 2
        return -a * numerator / denominator

    def jacobian_mu_p(self, r, z):
        """Find Jacobian of coordinate transformation from (r, z) to (mu_m, pm) at a point.

        Parameters
        ----------
        r: radius (m)
        z: height (m)
        Accepted as floats to compute for single point or numpy arrays to compute for many points.

        Returns
        -------
        Value of Jacobian at (r, z) according to Tailleux & Harris, Equation 3.25 (kg m / s^4).
        """
        jacobian = -(self.density(r, z) * self.g * (self.f ** 2) * (r ** 3) + self.dpsi_dr(r, z) *
                     self.pressure_vertical_gradient(r, z) - self.dpsi_dz(r, z) * self.pressure_radial_gradient(r, z))
        return jacobian

    def eddy_kinetic_energy_ratio(self, r, z):
        """Find ratio between eddy kinetic energy and mechanical eddy energy Pi_k at a point.

        Parameters
        ----------
        r: radius (m)
        z: height (m)
        Accepted as floats to compute for single point or numpy arrays to compute for many points.

        Returns
        -------
        Ratio of eddy kinetic energy (v-v_m)^2/2 to \Pi_k for small-amplitude perturbations:
        Tailleux & Harris Equations (3.37)/(3.36).
        """
        ratio = -(self.jacobian_mu_p(r, z) * r) / (4. * self.g * self.density(r, z) * self.angular_momentum(r, z) ** 2)
        return ratio

    def jacobian_ratio(self, r, z):
        """Find ratio of vortex Jacobian (Harris & Tailleux, Equation 3.25) to resting Jacobian at a point.

        Parameters
        ----------
        r: radius (m)
        z: height (m)
        Accepted as floats to compute for single point or numpy arrays to compute for many points.

        Returns
        -------
        Ratio of Jacobian to resting Jacobian.
        """
        resting_jacobian = -self.density(r, z) * self.g * (self.f**2) * (r**3)
        ratio = self.jacobian_mu_p(r, z) / resting_jacobian
        return ratio

    @staticmethod
    def chi(r):
        """
        Calculate chi = 1/(2r^2) from radius.
        Parameters
        ----------
        r: radius (m)
        Accepted as floats to compute for single radius or numpy arrays to compute for many radii.

        Returns
        -------
        chi (m^-2)
        """
        return 1./(2. * r**2)

    @classmethod
    def smith(cls):
        """Create vortex used by Smith (2005).

        Returns
        -------
        Vortex object with properties described in Appendix B of Smith (2005).
        """
        return cls(303., 2.12e-5, 95000., 5.e-5, 40000., 8000.)
