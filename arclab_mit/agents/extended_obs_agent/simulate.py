from astropy import units as u
from poliastro.twobody import Orbit
from poliastro.bodies import Body
import numpy as np
import time

kerbin_mu = 3.5316000e12 * (u.m**3 / u.s**2)
kerbin_radius = 600000 * u.m
kerbin_mass = 5.2915158e22 * u.kg
kerbin = Body(parent=None, k=kerbin_mu, name="Kerbin", R=kerbin_radius, mass=kerbin_mass)

def simulate(state, time):
    r_pursuer, v_pursuer, r_evader, v_evader = state
    r_pursuer *= u.m
    v_pursuer *= u.m / u.s
    r_evader *= u.m
    v_evader *= u.m / u.s
    time *= u.s

    pursuer_orbit = Orbit.from_vectors(kerbin, r_pursuer, v_pursuer)
    evader_orbit = Orbit.from_vectors(kerbin, r_evader, v_evader)

    pursuer_state = pursuer_orbit.propagate(time)
    evader_state = evader_orbit.propagate(time)

    return pursuer_state.r.to(u.m).value, pursuer_state.v.to(u.m / u.s).value, evader_state.r.to(u.m).value, evader_state.v.to(u.m / u.s).value

def closest_approach(state, max_time):
    min_distance = float('inf')
    closest_state = None
    closest_time = 0
    # calculate distance every second, and pick min
    for time in range(0, max_time+1, 1):
        future_state = simulate(state, time)
        r_pursuer, _, r_evader, _ = future_state
        distance = np.linalg.norm(np.array(r_pursuer) - np.array(r_evader))
        if distance < min_distance:
            min_distance = distance
            closest_state = future_state
            closest_time = time
    return closest_state, closest_time

# do one call to simulate to warm up (first call takes long). This doesn't affect calculations, only performance
r_pursuer = [708643.8192375809, -245609.31908026608, -0.0005146448301272027]
v_pursuer = [720.6227129174193, 2053.3207684576782, -0.00031452627985032464]
r_evader = [709510.9064440622, -243093.13780440498, -0.42475680700490526]
v_evader = [703.3420103326258, 2052.8297582878354, 0.0035828857958765214]
state = r_pursuer, v_pursuer, r_evader, v_evader
simulate(state, 10.0)

"""
start = time.time()
closest_approach(state, 300)
print(time.time() - start)
"""