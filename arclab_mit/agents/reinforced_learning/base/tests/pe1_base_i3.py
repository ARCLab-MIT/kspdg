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

        r0, r, tof = (
            state_dict["position"],
            state_dict["target_position"],
            self.burn_time * u.s,
        )

        k = 3.986004418e14 * u.m**3 / u.s**2

        v0, v = iod.lambert(k, r0, r, tof)

        commanded_velocity = v - state_dict["velocity"]

        thrust_direction = commanded_velocity / np.linalg.norm(commanded_velocity)

        action = np.zeros(4)
        action[0] = np.dot(thrust_direction, state_dict["position"])
        action[1] = np.dot(
            thrust_direction, np.cross([0, 0, 1], state_dict["position"])
        )
        action[2] = np.dot(thrust_direction, [0, 0, 1])
        action[3] = tof

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
