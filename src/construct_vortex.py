import numpy as np


f = 6.14e-5
g = 9.81
Rd = 287.
surface_temperature = 303.
lapse_rate = 2.12e-5
central_pressure = 95000.
far_field_pressure = 100000.
pressure_deficit = central_pressure - far_field_pressure
radial_scale = 20000.
radial_coefficient = 1.048
vertical_scale = 8000.
vertical_top = 16000.


def pressure_perturbation(r, z):
    radial_term = 1. - np.exp(-radial_coefficient*radial_scale/r)
    vertical_term = np.exp(-z/vertical_scale)*np.cos(0.5*np.pi*z/vertical_top)
    return pressure_deficit*radial_term*vertical_term


def environment_temperature(z):
    return surface_temperature * (1. - lapse_rate*z)


def environment_pressure(z):
    return far_field_pressure * np.exp((g*np.log(1. - lapse_rate*z))/(Rd*surface_temperature*lapse_rate))


def environment_pressure_vertical_gradient(z):
    return -(far_field_pressure*g*(1. - lapse_rate*z)**(g/(Rd*surface_temperature*lapse_rate) - 1.))/(Rd*surface_temperature)


def pressure(r, z):
    return environment_pressure(z) + pressure_perturbation(r, z)


def pressure_vertical_gradient(r, z):
    radial_term = 1. - np.exp(-radial_coefficient*radial_scale/r)
    vertical_term = -np.exp(-z / vertical_scale) * (1. / vertical_scale) * (
                (0.5 * np.pi * vertical_scale / vertical_top) * np.sin(0.5 * np.pi * z / vertical_top) + np.cos(
            0.5 * np.pi * z / vertical_top))
    return pressure_deficit*radial_term*vertical_term + environment_pressure_vertical_gradient(z)


def pressure_radial_gradient(r, z):
    radial_term = -radial_coefficient*radial_scale*np.exp(-radial_coefficient*radial_scale/r)/(r**2)
    vertical_term = np.exp(-z / vertical_scale) * np.cos(0.5 * np.pi * z / vertical_top)
    return pressure_deficit * radial_term * vertical_term


def density(r, z):
    return -pressure_vertical_gradient(r, z)/g


def temperature(r, z):
    return pressure(r, z)/(Rd * density(r, z))


def gradient_wind_term(r, z):
    #fv + v^2/r
    return pressure_radial_gradient(r, z)/density(r, z)


def azimuthal_wind(r, z):
    gradient_wind = gradient_wind_term(r, z)
    v = 0.5*r*(-f + np.sqrt(f**2 + 4*gradient_wind/r))
    return v
