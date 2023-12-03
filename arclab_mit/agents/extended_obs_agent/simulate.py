from astropy import units as u
from poliastro.twobody import Orbit
from poliastro.bodies import Body
import numpy as np

kerbin_mu = 3.5316000e12 * (u.m**3 / u.s**2)
kerbin_radius = 600000 * u.m
kerbin_mass = 5.2915158e22 * u.kg
kerbin = Body(parent=None, k=kerbin_mu, name="Kerbin", R=kerbin_radius, mass=kerbin_mass)

def simulate(r, v, time):
    r *= u.m
    v *= u.m / u.s
    time *= u.s
    orbit = Orbit.from_vectors(kerbin, r, v)
    future_state = orbit.propagate(time)
    return future_state.r.to(u.m).value, future_state.v.to(u.m / u.s).value

def closest_approach(state, max_time):
    r1, v1, r2, v2 = state
    min_distance = float('inf')
    closest_state = None
    closest_time = 0
    # calculate distance every second, and pick min
    for time in range(0, max_time+1, 1):
        r1_future, v1_future = simulate(r1, v1, time)
        r2_future, v2_future = simulate(r2, v2, time)
        distance = np.linalg.norm(np.array(r1_future) - np.array(r2_future))
        if distance < min_distance:
            min_distance = distance
            closest_state = (r1_future, v1_future, r2_future, v2_future)
            closest_time = time
    return closest_state, closest_time

# do one call to simulate to warm up (first call takes long). This doesn't affect calculations, only performance
r = [708643.8192375809, -245609.31908026608, -0.0005146448301272027]
v = [720.6227129174193, 2053.3207684576782, -0.00031452627985032464]
simulate(r, v, 10.0)