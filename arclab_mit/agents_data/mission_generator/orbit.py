import os
import math

import numpy as np
import csv
import datetime
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.bodies import Body
from poliastro.constants import J2000
from poliastro.plotting.static import StaticOrbitPlotter
from poliastro.plotting.interactive import OrbitPlotter3D
from poliastro.frames import Planes

from scipy.spatial.transform import Rotation

# CONSTANTS
kerbin_mu = 3.5316000e12 << (u.m ** 3 / u.s ** 2)   # Gravitational parameter of Kerbin
kerbin_radius = 600000 << u.m                       # Radius of Kerbin
kerbin_mass = 5.2915158e22 << u.kg                  # Mass of Kerbin
kerbin = Body(parent=None, k=kerbin_mu, name="Kerbin", R=kerbin_radius, mass=kerbin_mass)

MAX_ECC = 0.1 << u.one  # Maximum eccentricity
MAX_INC = 3 << u.deg   # Maximum inclination

class MyOrbit:
    """ Class representing a keplerian orbit
    """
    def __init__(self, *args):
        """ Epoch time is 2000-01-01 12:00:00 UTC. It is not relevant since any value could be used.
        """
        epoch = J2000
        if len(args) == 2:
            self.r = args[0]
            self.v = args[1]
            self.orb = Orbit.from_vectors(kerbin, self.r, self.v, epoch=epoch)

            # Extract Keplerian elements
            sma, ecc, inc, lan, lpe, mna = self.orb.classical()
            self.sma = sma << u.m
            self.ecc = ecc
            self.inc = inc
            self.lpe = lpe
            self.lan = lan
            self.mna = mna << u.rad
        else:
            self.sma = args[0]
            self.ecc = args[1]
            self.inc = args[2]
            self.lpe = args[3]
            self.lan = args[4]
            self.mna = args[5]
            self.orb = Orbit.from_classical(kerbin, self.sma, self.ecc, self.inc, self.lan, self.lpe, self.mna, epoch=epoch)

            # Extract position and velocity
            self.r, self.v = self.orb.rv()

    """ Retrieve keplerian elements (orbital parameters)
    """
    def get_keplerian_elements(self):
        sma = self.sma.to(u.m).value
        ecc = self.ecc.value
        inc = self.inc.to(u.deg).value
        lpe = self.lpe.to(u.deg).value
        lan = self.lan.to(u.deg).value
        mna = self.mna.to(u.rad).value
        return [sma, ecc, inc, lpe, lan, mna]


def orbital_from_vectors(mu, r, v):
    """ Obtain orbital parameters from position / velocity vector
        Args:
            mu: gravitational parameter of the orbiting body
            r: position vector
            v: velocity vector
    """

    # radius
    rabs = np.linalg.norm(r, ord=2)

    # vr: radial velocity
    vr = np.dot(r, v) / rabs

    # h: angular momentum
    h = np.cross(r, v)
    habs = np.linalg.norm(h, ord=2)

    # inclination
    inc = np.arccos(h[2]/habs)

    # longitude of ascending node
    n = np.cross(np.array([0, 0, 1]), h)
    nabs = np.linalg.norm(n, ord=2)
    lan = np.arccos(n[0]/nabs)
    if n[1] < 0:
        lan = (np.pi*2 << u.rad) - lan

    # eccentricity
    e = 1 / mu * (np.cross(v, h) - mu * r/rabs)
    eabs = np.linalg.norm(e, ord=2)
    ecc = eabs

    # argument of periapsis
    lpe = np.arccos(np.dot(n, e) / nabs / ecc)
    if e[2] < 0:
        lpe = (np.pi*2 << u.rad) - lpe

    # true anomaly
    mna = np.arccos(np.dot(e, r)/eabs/rabs)
    if vr < 0:
        mna = (np.pi*2 << u.rad) - mna

    # Obtain semi-major axis from angular momentum and eccentricity
    sma = habs**2 / (mu * (1 - ecc**2))

    return [sma, ecc, inc, lpe, lan, mna]


def vector_from_orbitals (mu, sma, ecc, inc, lpe, lan, mna):
    """ Obtain position / velocity vector from orbital parameters
        Args:
            mu: gravitational parameter of the orbiting body
            sma: semi-major axis
            ecc: eccentricity
            inc: inclination
            lpe: argument of periapsis
            lan: longitude of ascending node
            mna: true anomaly
    """

    inc = inc.to_value(u.rad)
    lpe = lpe.to_value(u.rad)
    lan = lan.to_value(u.rad)

    # Compute angular momentum
    h = np.sqrt(mu * sma * (1 - ecc**2))

    # Transform to perifocal frame
    r_w = h**2 / mu / (1 + ecc * np.cos(mna)) * np.array((np.cos(mna), np.sin(mna), 0))
    v_w = mu / h * np.array((-np.sin(mna), ecc + np.cos(mna), 0))

    # Rotate to transform from perifocal frame to inertial frame
    R = Rotation.from_euler("ZXZ", [-lpe, -inc, -lan])
    r_rot = r_w @ R.as_matrix()
    v_rot = v_w @ R.as_matrix()
    return [r_rot, v_rot]



def sample_orbit(evader_orbit, dmin, dmax, speed_range, circular=True, precise=False):
    """ Ârgs
        evader_orbit: evader's orbit
        dmin: minimum distance between evader and pursuer
        dmax: maximum distance between evader and pursuer
        circular: if True, pursuer's orbit is circular
        precise: if True, pursuer's orbit is generated precisely

    Returns a keplerian orbit for the pursuer

    Algorithm:
    1. Sample a random distance between dmin and dmax
    2. If precise is True, pursuer's initial position is appoximately [dmin, dmax] apart from evader's position
        - if circular is True, pursuer's initial velocity is perpendicular to the radial direction
        - otherwise pursuer's initial velocity has magnitude in range (1-speed_range/2) (1+speed_range/2) and a random direction
    3. If precise is False, split distance along the radial and orbital direction
        - Force initial position to become the ascending node of the orbit as follows:
            - Match lan with evader's mna since evader's inclination is approximately 0
        - Sample random semi-major axis in range [evader_orbit.sma-d_rad, evader_orbit.sma+d_rad]
        - Sample delta mna in range [-d_orb/sma, d_orb/sma]
        - Sample random eccentricity in range [0, MAX_ECC]
        - Sample random inclination in range [0, MAX_INC] deg

    NOTE: Best results for default circular (True) and precise (False).
    """

    """ Sample a random distance
    """
    d = dmin + np.random.rand(1)[0] * (dmax - dmin)
    d *= u.m
    if precise:
        """ Pursuer's initial position is appoximately [dmin, dmax] apart from evader's position
            """
        t = np.random.rand(3)
        t = t / np.linalg.norm(t)
        r_vec = evader_orbit.r + d * t

        if circular:
            """ Find orbital speed at given position
            """
            r = np.linalg.norm(r_vec, ord=2)
            speed = np.sqrt(kerbin_mu.to_value(u.m**3/u.s**2)/r.to_value(u.m))
            speed = (speed << u.m/u.s) << u.km/u.s

            """ Generate a random velocity direction vector perpendicular to r_vec
            """
            t = np.random.rand(3)
            t = np.cross(r_vec.value, t)
            t = t / np.linalg.norm(t, ord=2)
            v_vec = speed * t
        else:
            """ Pursuer's initial velocity has magnitude in range (1-speed_range/2) (1+speed_range/2)
                and a random direction.
            """
            t = np.random.rand(1)[0] - 0.5
            speed = np.linalg.norm(evader_orbit.v) * t
            t = np.random.rand(3)
            t = t / np.linalg.norm(t)
            v_vec = speed * t
        pursuer_orbit = MyOrbit(r_vec, v_vec)

        """ Code to check poliastro calculations
        orbital_elements = orbital_from_vectors (kerbin_mu << u.km ** 3 / u.s ** 2,
                                                 r_vec << u.km,
                                                 v_vec << u.km / u.s)
        vectors = vector_from_orbitals(kerbin_mu << u.km ** 3 / u.s ** 2,
                                      orbital_elements[0],
                                      orbital_elements[1],
                                      orbital_elements[2],
                                      orbital_elements[3],
                                      orbital_elements[4],
                                      orbital_elements[5]
                                      )
        """
    else:
        """ Split distance along the radial and orbital direction
        """
        frac = np.random.rand(1)[0]
        d_rad = d * frac
        if np.random.rand(1)[0] < 0.5:
            d_rad = -d_rad
        d_orb = d * (1 - frac)
        if np.random.rand(1)[0] < 0.5:
            d_orb = -d_orb

        """ Determine ecc and sma
        """
        if circular:
            ecc = 0 << u.one
            sma = evader_orbit.sma + d_rad
        else:
            frac = np.random.rand(1)[0]
            ecc = frac * MAX_ECC
            # r is given by the equation r = a(1-e^2)/(1+e*cos(nu)) where a is sma and nu the true anomaly.
            # r should be close to evader_orbit.sma
            sma = evader_orbit.sma / ((1 - ecc**2) / (1 + ecc * np.cos(evader_orbit.mna.to_value(u.rad))))
            sma += d_rad
        """ Sample random inclination in [-MAX_INC, MAX_INC] deg
        """
        inc = np.random.rand(1)[0] * MAX_INC
        if (np.random.rand(1)[0] < 0.5):
            inc = (180<<u.degree) - inc
        lpe = evader_orbit.lpe
        """ Match lan with evader's sna since evader's inclination is approximately 0
        """
        lan = evader_orbit.mna
        delta = d_orb.to_value(u.km) / evader_orbit.sma.to_value(u.km)
        delta = delta << u.rad
        mna = delta
        mna = 0 << u.rad
        pursuer_orbit = MyOrbit(sma, ecc, inc, lpe, lan, mna)

    return pursuer_orbit


def sample_n_orbits(n, dmin, dmax, speed_range):
    orbits = []

    print("Generating orbits...")
    print("\n\n *** VALUES: ", n, dmin, dmax, speed_range, "\n\n")

    evader_orbit = MyOrbit(750000 << u.m,
                           0 << u.one,
                           0.0001 << u.deg,
                           0 << u.deg,
                           0 << u.deg,
                           5.9341194567807207 << u.rad)

    while n > 0:
        pursuer_orbit = sample_orbit(evader_orbit, dmin, dmax, speed_range, circular=False, precise=False)

        """ Accept orbit if ecc < MAX_ECC and inc << MAX_INC
        """
        if pursuer_orbit.ecc < MAX_ECC and pursuer_orbit.inc < MAX_INC:
            d = np.linalg.norm(evader_orbit.r - pursuer_orbit.r)
            if (d.to(u.m).value > dmin) and (d.to(u.m).value < dmax):
                n -= 1
                orbits.append(pursuer_orbit)

                speed = np.linalg.norm(evader_orbit.v - pursuer_orbit.v)
                print(f" Distance from evader: {d:.5f}")
                print(f" Relative speed: {speed:.5f}")
                print(f" Pursuer position: {pursuer_orbit.r}")

    return orbits


if __name__ == '__main__':
    test_orbit = MyOrbit(749999.9999999994 << u.m,
                           0.08261906862959273 << u.one,
                           0.0001 << u.deg,
                           180 << u.deg,
                           0 << u.deg,
                           2.7743753789701864 << u.rad)
    print(f"Test position: {test_orbit.r}")

    dmin = float(input("Min distance: "))
    dmax = float(input("Max distance: "))
    speed_range = float(input("Speed range in %: ")) / 100

    evader_orbit = MyOrbit(750000 << u.m,
                           0 << u.one,
                           0.0001 << u.deg,
                           0 << u.deg,
                           0 << u.deg,
                           5.9341194567807207 << u.rad)

    n = int(input("Number of orbits: "))

    print("")
    print("Evader's information")
    print(f" Evader's altitude: {np.linalg.norm(evader_orbit.r):.5f}")
    print(f" Evader's speed: {np.linalg.norm(evader_orbit.v):.5f}")
    print("")

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_name = "./logs/orbit_" + timestamp + '.csv'

    log_directory = 'logs'
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    log = open(log_name, mode='w', newline='')
    head = ['e_sma', 'e_ecc', 'e_inc', 'e_lpe', 'e_lan', 'e_mna',
            'p_sma', 'p_ecc', 'p_inc', 'p_lpe', 'p_lan', 'p_mna',
            'dist', 'speed']
    csv.writer(log).writerow(head)

    # Init orbit plotter
    figure = plt.figure()
    #    plotter = StaticOrbitPlotter()
    plotter = OrbitPlotter3D()
    plotter.set_attractor(kerbin)
    plotter.plot(evader_orbit.orb, label="evader", color="red")

    while n > 0:
        pursuer_orbit = sample_orbit(evader_orbit, dmin, dmax, speed_range)

        """ Accept orbit if ecc < MAX_ECC and inc << MAX_INC
        """
        if pursuer_orbit.ecc < MAX_ECC and pursuer_orbit.inc < MAX_INC:
            n -= 1
            # Print Keplerian elements
            print("Pursuer's orbit:")
            print(" Semi-major axis (sma):", pursuer_orbit.sma)
            print(" Eccentricity (ecc):", pursuer_orbit.ecc)
            print(" Inclination (inc):", pursuer_orbit.inc)
            print(" Right Ascension of the Ascending Node (lan):", pursuer_orbit.lan)
            print(" Argument of Perigee (lpe):", pursuer_orbit.lpe)
            print(" True Anomaly (mna):", pursuer_orbit.mna)
            print("")
            d = np.linalg.norm(evader_orbit.r - pursuer_orbit.r)
            speed = np.linalg.norm(evader_orbit.v - pursuer_orbit.v)
            print(f" Distance from evader: {d:.5f}")
            print(f" Relative speed: {speed:.5f}")
            print("")

            # Plot orbit
            plotter.plot(pursuer_orbit.orb, label="pursuer", color="blue")

            # Save orbits
            row = evader_orbit.get_keplerian_elements() + pursuer_orbit.get_keplerian_elements()
            row += [d.to(u.m).value, speed.to(u.m / u.s).value]
            csv.writer(log).writerow(row)

    log.close()

    print(f"Evader position: {evader_orbit.r}")
    print(f"Pursuer position: {pursuer_orbit.r}")

    fig_name = "./logs/orbit_" + timestamp + '.html'
    plotter._figure.write_html(fig_name, auto_open=True)
    # fig_name = "./logs/orbit_" + timestamp + '.png'
    # plotter._figure.write_image(fig_name, format='png')
    plt.close()
