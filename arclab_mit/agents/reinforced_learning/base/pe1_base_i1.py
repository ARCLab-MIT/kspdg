from kspdg.pe1.e1_envs import PE1_E1_I3_Env
import numpy as np
from kspdg.agent_api.base_agent import KSPDGBaseAgent
from kspdg.agent_api.runner import AgentEnvRunner


class NaivePursuitAgent(KSPDGBaseAgent):
    """An agent that naively burns directly toward it's target"""

    def __init__(self):
        super().__init__()

    def get_action(self, observation, gain=3.0):
        print(observation)
        state_dict = {
            "position": observation[3:6],
            "velocity": observation[6:9],
        }
        position = np.array(state_dict["position"])
        velocity = np.array(state_dict["velocity"])

        relative_velocity = -velocity / np.linalg.norm(velocity)

        if np.linalg.norm(position) > 0:  # if not yet reached
            line_of_sight_rate = (
                np.cross(position, velocity) / np.linalg.norm(position) ** 2
            )
            commanded_acceleration = (
                gain * np.linalg.norm(velocity) * line_of_sight_rate
            )
        else:
            commanded_acceleration = np.array([0, 0, 0])

        action = np.zeros(4)
        if commanded_acceleration[0] > 0:
            action[0] = 1
        elif commanded_acceleration[0] < 0:
            action[1] = 1
        if commanded_acceleration[1] > 0:
            action[2] = 1
        elif commanded_acceleration[1] < 0:
            action[3] = 1

        return action


if __name__ == "__main__":
    naive_agent = NaivePursuitAgent()
    runner = AgentEnvRunner(
        agent=naive_agent,
        env_cls=PE1_E1_I3_Env,
        env_kwargs=None,
        runner_timeout=100,  # agent runner that will timeout after 100 seconds
        debug=True,
    )
    print(runner.run())
