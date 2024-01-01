from kspdg.agent_api.base_agent import KSPDGBaseAgent
from kspdg.pe1.e1_envs import PE1_E1_I3_Env
from kspdg.agent_api.runner import AgentEnvRunner
import numpy as np


class ProportionalNavigationGuidanceAgent(KSPDGBaseAgent):
    """An agent that uses Proportional Navigation Guidance to target its objective and dynamically adjusts throttle."""

    def __init__(self, N=3, thresh_distance=500, speed_limit=100, decel_distance=1000):
        super().__init__()
        self.N = N
        self.prev_target_position = None
        self.thresh_distance = thresh_distance
        self.speed_limit = speed_limit
        self.decel_distance = decel_distance

    def get_action(self, observation):
        # unpack target and pursuer positions and velocities from observation list
        pursuer_position = np.array(observation[3:6])
        target_position = np.array(observation[9:12])
        pursuer_velocity = np.array(observation[6:9])
        target_velocity = np.array(observation[12:15])

        if self.prev_target_position is not None:
            relative_velocity = target_position - self.prev_target_position
            relative_velocity_agent = pursuer_velocity - target_velocity

            line_of_sight_rate = (
                np.cross(target_position, relative_velocity)
                / np.linalg.norm(target_position) ** 2
            )
            required_acceleration = (
                self.N * np.linalg.norm(relative_velocity_agent) * line_of_sight_rate
            )

            current_distance = np.linalg.norm(target_position - pursuer_position)
            throttle = max(
                current_distance / self.thresh_distance, 0.1
            )  # impose a lower limit to the throttle

            if (
                current_distance <= self.decel_distance
            ):  # a rule for the spaceship to decelerate as it approaches target
                throttle = 0.1
            output = [*required_acceleration, throttle]
            print(output)
            return output

        self.prev_target_position = target_position
        print([0, 0, 0, 0])
        return [0, 0, 0, 0]


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
