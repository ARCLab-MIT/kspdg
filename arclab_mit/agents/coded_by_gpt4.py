from kspdg.agent_api.base_agent import KSPDGBaseAgent
from kspdg.pe1.e1_envs import PE1_E1_I3_Env
from kspdg.agent_api.runner import AgentEnvRunner


from poliastro.twobody import Orbit
from poliastro.maneuver import Maneuver
from poliastro.bodies import Body
from astropy import units as u
from collections import deque
import numpy as np

# Define Kerbin
class KERBIN:
    G = 9.80665 # standard gravity [m/s/s]
    RADIUS = 6.0e5 * u.m
    MU = 3.5316e12 * u.m**3 / u.s**2
    # Mass is derived from MU = G * Mass
    MASS = MU / G

Kerbin = Body(parent=None, name="Kerbin", k=KERBIN.MU, R=KERBIN.RADIUS, mass=KERBIN.MASS)

class CodedByGPT4Agent(KSPDGBaseAgent):

    def __init__(self, history_size=5):
        super().__init__()
        # Constants (from your provided data)
        self.VACUUM_MAX_THRUST_FORWARD = 8000
        self.VACUUM_MAX_THRUST_RIGHT = 4000
        self.VACUUM_MAX_THRUST_DOWN = 4000
        # History buffer
        self.history = deque(maxlen=history_size)

    def get_orbit_from_observation(self, observation):
        r = observation[3:6] * u.m
        v = observation[6:9] * u.m / u.s
        return Orbit.from_vectors(Kerbin, r, v)

    def get_action(self, observation):
        # Append the current observation to history
        self.history.append(observation)
        print("Observation:")
        print(observation)
        
        # Use history to derive more info (like average velocities, accelerations, etc.)
        # For demonstration, let's just use the current observation:
        current_observation = self.history[-1]
        
        pursuer_orbit = self.get_orbit_from_observation(current_observation)
        evader_orbit = self.get_orbit_from_observation(current_observation[0:3] + current_observation[9:15])

        # Compute relative position and velocity
        rel_pos = (evader_orbit.r - pursuer_orbit.r).value
        rel_vel = (evader_orbit.v - pursuer_orbit.v).value

        # For now, let's make dV align with the relative velocity
        dV_direction = rel_vel / np.linalg.norm(rel_vel)
        # The magnitude of dV can be set to a fraction of the max thrust, for instance:
        dV_magnitude = 0.5 * self.VACUUM_MAX_THRUST_FORWARD / current_observation[1]  # This can be refined further

        dV = dV_direction * dV_magnitude

        # Normalize desired dV with respect to maximum possible delta-V
        throttle_forward = dV[0] / (self.VACUUM_MAX_THRUST_FORWARD / current_observation[1])
        throttle_right = dV[1] / (self.VACUUM_MAX_THRUST_RIGHT / current_observation[1])
        throttle_down = dV[2] / (self.VACUUM_MAX_THRUST_DOWN / current_observation[1])

        # Clip values
        throttle_forward = np.clip(throttle_forward, -1.0, 1.0)
        throttle_right = np.clip(throttle_right, -1.0, 1.0)
        throttle_down = np.clip(throttle_down, -1.0, 1.0)

        # Burn duration (can be adjusted based on your strategy)
        burn_duration = 1.0
        action = [throttle_forward, throttle_right, throttle_down, burn_duration]
        print("Action:")
        print(action)
        return action


if __name__ == "__main__":
    
    my_agent = CodedByGPT4Agent()    
    runner = AgentEnvRunner(
        agent=my_agent, 
        env_cls=PE1_E1_I3_Env, 
        env_kwargs=None,
        runner_timeout=240,     # agent runner that will timeout after 100 seconds
        debug=False)
    runner.run()
