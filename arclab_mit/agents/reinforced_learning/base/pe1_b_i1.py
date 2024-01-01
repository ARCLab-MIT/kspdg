from kspdg.agent_api.base_agent import KSPDGBaseAgent
from kspdg.pe1.e1_envs import PE1_E1_I3_Env
from kspdg.agent_api.runner import AgentEnvRunner
import numpy as np


class ProportionalNavigationGuidanceAgent(KSPDGBaseAgent):
    """An agent that uses Proportional Navigation Guidance to target its objective"""

    def __init__(self, N=3):
        super().__init__()
        self.N = N
        self.prev_target_position = None

    def get_action(self, observation):
        """compute agent's action given observation"""

        # unpack target and pursuer positions from observation list
        pursuer_position = np.array(observation[3:6])
        target_position = np.array(observation[9:12])

        if self.prev_target_position is not None:
            # calculate the relative velocity and determine necessary acceleration
            relative_velocity = target_position - self.prev_target_position

            line_of_sight_rate = (
                np.cross(target_position, relative_velocity)
                / np.linalg.norm(target_position) ** 2
            )
            required_acceleration = (
                self.N * np.linalg.norm(relative_velocity) * line_of_sight_rate
            )

            return {
                "burn_vec": [
                    *required_acceleration,
                    1.0,
                ],  # throttle in x-axis, throttle in y-axis, throttle in z-axis, duration [s]
                "ref_frame": 0,  # burn_vec expressed in agent vessel's right-handed body frame.
            }

        self.prev_target_position = target_position
        return {"burn_vec": [0, 0, 0, 0], "ref_frame": 0}


if __name__ == "__main__":
    png_agent = ProportionalNavigationGuidanceAgent(N=3)
    runner = AgentEnvRunner(
        agent=png_agent,
        env_cls=PE1_E1_I3_Env,
        env_kwargs=None,
        runner_timeout=100,
        debug=True,
    )
    print(runner.run())
