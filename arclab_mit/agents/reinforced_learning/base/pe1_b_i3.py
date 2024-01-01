from kspdg.agent_api.base_agent import KSPDGBaseAgent
from kspdg.pe1.e1_envs import PE1_E1_I3_Env
from kspdg.agent_api.runner import AgentEnvRunner
import numpy as np


class ProportionalNavigationGuidanceAgent(KSPDGBaseAgent):
    """An agent that uses Proportional Navigation Guidance to target its objective and dynamically adjusts throttle."""

    def __init__(self, N=2.5, thresh_distance=500, speed_limit=100):
        super().__init__()
        self.N = N
        self.prev_target_position = None
        self.thresh_distance = thresh_distance
        self.speed_limit = speed_limit

    def get_action(self, observation):
        # unpack target and pursuer positions from observation list
        pursuer_position = np.array(observation[3:6])
        target_position = np.array(observation[9:12])
        pursuer_velocity = np.array(observation[6:9])

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

            # Adjust throttle based on distance to target
            current_distance = np.linalg.norm(target_position - pursuer_position)
            throttle = max(
                current_distance / self.thresh_distance, 0.1
            )  # lower limit throttle to 10%

            # Add speed limit to manage fuel efficiency
            current_speed = np.linalg.norm(pursuer_velocity)
            if current_speed > self.speed_limit:
                throttle = 0

            return {"burn_vec": [*required_acceleration, throttle], "ref_frame": 0}

        self.prev_target_position = target_position
        return {"burn_vec": [0, 0, 0, 0], "ref_frame": 0}


if __name__ == "__main__":
    png_agent = ProportionalNavigationGuidanceAgent(N=2.5)
    runner = AgentEnvRunner(
        agent=png_agent,
        env_cls=PE1_E1_I3_Env,
        env_kwargs=None,
        runner_timeout=100,
        debug=True,
    )
    print(runner.run())
