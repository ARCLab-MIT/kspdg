import os

import numpy as np
import csv
import datetime
import matplotlib.pyplot as plt

from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.bodies import Body
from poliastro.plotting.static import StaticOrbitPlotter
from poliastro.plotting.interactive import OrbitPlotter3D

kerbin_mu = 3.5316000e12 << (u.m**3 / u.s**2)
kerbin_radius = 600000 << u.m
kerbin_mass = 5.2915158e22 << u.kg
kerbin = Body(parent=None, k=kerbin_mu, name="Kerbin", R=kerbin_radius, mass=kerbin_mass)

class MyOrbit:
    def __init__(self, * args):
        if len(args) == 2:
            self.r = args[0]
            self.v = args[1]
            self.orb = Orbit.from_vectors(kerbin, self.r, self.v)
            # Extract Keplerian elements
            sma, ecc, inc, lpe, lan, mna = self.orb.classical()
            self.sma = sma
            self.ecc = ecc
            self.inc = inc
            self.lpe = lpe
            self.lan = lan
            self.mna = mna
        else:
            self.sma = args[0]
            self.ecc = args[1]
            self.inc = args[2]
            self.lpe = args[3]
            self.lan = args[4]
            self.mna = args[5]
            self.orb = Orbit.from_classical(kerbin, self.sma, self.ecc, self.inc, self.lpe, self.lan, self.mna)
            # Extract position and velocity
            self.r, self.v = self.orb.rv()

    def get_keplerian_elements(self):
        sma = self.sma.to(u.m).value
        ecc = self.ecc.value
        inc = self.inc.to(u.rad).value
        lpe = self.lpe.to(u.deg).value
        lan = self.lan.to(u.deg).value
        mna = self.lan.to(u.rad).value
        return [sma, ecc, inc, lpe, lan, mna]


def sample_orbit (evader_orbit, dmin, dmax, speed_range):
    """ Pursuer's initial position is [dmin, dmax] apart from evader's position
    """
    d = dmin + np.random.rand(1)[0] * (dmax - dmin)
    d *= u.m
    t = np.random.rand(3)
    t = t / np.linalg.norm(t)
    r_vec = evader_orbit.r + d * t

    """ Pursuer's initial velocity has magnitude in range (1-speed_range/2) (1+speed_range/2)
        and a random direction.
    """
    t = np.random.rand(1)[0] - 0.5
    speed = np.linalg.norm(evader_orbit.v)*t
    t = np.random.rand(3)
    t = t / np.linalg.norm(t)
    v_vec = np.linalg.norm(evader_orbit.v)*t

    pursuer_orbit = MyOrbit(r_vec, v_vec)
    return pursuer_orbit


if __name__ == '__main__':
    dmin = float(input("Min distance: "))
    dmax = float(input("Max distance: "))
    speed_range = float(input("Speed range in %: "))/100

    evader_orbit = MyOrbit(750000 << u.km,
                           0 << u.one,
                           0.0001 << u.rad,
                           0 << u.rad,
                           0 << u.rad,
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

        """ Accept orbit if ecc < 0.1
        """
        if pursuer_orbit.ecc < 0.1:
            n -= 1
            # Print Keplerian elements
            print("Pursuer's orbit:")
            print(" Semi-major axis (sma):", pursuer_orbit.sma)
            print(" Eccentricity (ecc):", pursuer_orbit.ecc)
            print(" Inclination (inc):", pursuer_orbit.inc)
            print(" Right Ascension of the Ascending Node (lpe):", pursuer_orbit.lpe)
            print(" Argument of Perigee (lan):", pursuer_orbit.lan)
            print(" True Anomaly (mna):", pursuer_orbit.mna)
            print("")
            d = np.linalg.norm(evader_orbit.r - pursuer_orbit.r)
            speed = np.linalg.norm(evader_orbit.v - pursuer_orbit.v)
            print (f" Distance from evader: {d:.5f}")
            print (f" Relative speed: {speed:.5f}")
            print("")

            # Plot orbit
            plotter.plot(pursuer_orbit.orb, label="pursuer", color="blue")

            # Save orbits
            row = evader_orbit.get_keplerian_elements() + pursuer_orbit.get_keplerian_elements()
            row += [d.to(u.m).value, speed.to(u.m/u.s).value]
            csv.writer(log).writerow(row)

    log.close()

    fig_name = "./logs/orbit_" + timestamp + '.html'
    plotter._figure.write_html(fig_name, auto_open=True)
    # fig_name = "./logs/orbit_" + timestamp + '.png'
    # plotter._figure.write_image(fig_name, format='png')
    plt.close()
