from poliastro import iod
from astropy import units as u
from kspdg.pe1.e1_envs import PE1_E1_I3_Env
import numpy as np
from kspdg.agent_api.base_agent import KSPDGBaseAgent
from kspdg.agent_api.runner import AgentEnvRunner


class PoliastroAgent(KSPDGBaseAgent):
    """An agent that uses poliastro to solve Lambert's problem and follow the computed orbit"""

    def __init__(self):
        super().__init__()
        self.burn_time = 5  # Set a burn time

    def get_action(self, observation):
        position_m = observation[3:6] * u.m
        velocity_m_per_s = observation[6:9] * (u.m / u.s)
        target_position_m = observation[9:12] * u.m
        target_velocity_m_per_s = observation[12:15] * (u.m / u.s)

        state_dict = {
            "position": position_m,
            "velocity": velocity_m_per_s,
            "target_position": target_position_m,
            "target_velocity": target_velocity_m_per_s,
        }

        # Solve Lambert's problem to get the transfer orbit
        r0, r, tof = (
            state_dict["position"],
            state_dict["target_position"],
            self.burn_time * u.s,
        )

        k = 3.986004418e14 * u.m**3 / u.s**2  # Define k in m^3/s^2
        # k = k.to(u.km**3 / u.s**2)  # Convert k to km^3/s^2

        # Now use k in the iod.lambert function
        v0, v = iod.lambert(k, r0, r, tof)

        commanded_velocity = v - state_dict["velocity"]

        action = np.zeros(4)
        if commanded_velocity[0] > 0:
            action[0] = 1
        elif commanded_velocity[0] < 0:
            action[1] = 1
        if commanded_velocity[1] > 0:
            action[2] = 1
        elif commanded_velocity[1] < 0:
            action[3] = 1
        print(action, "action vector")
        print(observation, "observation")
        return action


if __name__ == "__main__":
    poliastro_agent = PoliastroAgent()
    runner = AgentEnvRunner(
        agent=poliastro_agent,
        env_cls=PE1_E1_I3_Env,
        env_kwargs=None,
        runner_timeout=100,
        debug=True,
    )
    print(runner.run())
