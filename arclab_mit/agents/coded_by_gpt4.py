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
    """
    Coded by GPT-4 after this conversation
    https://chat.openai.com/share/f1919fba-e5b2-474e-9c2e-39f878305701
    """

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
        self.history.append(np.array(observation))
        print("Current observation:")
        print(observation)
        window_size = min(5, len(self.history))  # Last 5 observations
        # Convert deque to numpy array for slicing and averaging
        avg_observation = np.mean(np.array(list(self.history))[-window_size:], axis=0)

        pursuer_orbit = self.get_orbit_from_observation(avg_observation)
        evader_orbit = self.get_orbit_from_observation(np.concatenate([avg_observation[0:3], avg_observation[9:15]]))

        rel_pos = (evader_orbit.r - pursuer_orbit.r).value
        rel_vel = (evader_orbit.v - pursuer_orbit.v).value

        dV_direction = rel_vel / np.linalg.norm(rel_vel)

        distance_to_target = np.linalg.norm(rel_pos)
        dV_magnitude = distance_to_target * self.VACUUM_MAX_THRUST_FORWARD / avg_observation[1]

        dV = dV_direction * dV_magnitude

        throttle_forward = dV[0] / (self.VACUUM_MAX_THRUST_FORWARD / avg_observation[1])
        throttle_right = dV[1] / (self.VACUUM_MAX_THRUST_RIGHT / avg_observation[1])
        throttle_down = dV[2] / (self.VACUUM_MAX_THRUST_DOWN / avg_observation[1])

        throttle_forward = np.clip(throttle_forward, -1.0, 1.0)
        throttle_right = np.clip(throttle_right, -1.0, 1.0)
        throttle_down = np.clip(throttle_down, -1.0, 1.0)

        burn_duration = 1.0  # This could also be dynamically adjusted
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
