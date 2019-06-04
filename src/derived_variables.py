from construct_vortex import *


p0 = 100000.
cp = 1004.5


def angular_momentum(r, z):
    return np.sqrt(r**3 * (gradient_wind_term(r, z) + 0.25 * r * f**2))


def potential_temperature(r, z):
    return temperature(r, z) * (p0/pressure(r ,z))**(Rd/cp)


def entropy(r, z):
    return cp * np.log(potential_temperature(r, z))


def geopotential(z):
    return g * z
