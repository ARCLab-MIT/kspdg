from kspdg.pe1.e1_envs import PE1_E1_I3_Env
import numpy as np
from kspdg.agent_api.base_agent import KSPDGBaseAgent
from kspdg.agent_api.runner import AgentEnvRunner


class NaivePursuitAgent(KSPDGBaseAgent):
    """An agent that naively burns directly toward it's target"""

    def __init__(self):
        super().__init__()

    def get_action(self, state, gain=3.0):
        position = np.array(state["position"])
        velocity = np.array(state["velocity"])

        relative_velocity = -velocity / np.linalg.norm(velocity)

        if np.linalg.norm(position) > 0:  # if not yet reached
            line_of_sight_rate = (
                np.cross(position, velocity) / np.linalg.norm(position) ** 2
            )
            commanded_acceleration = (
                gain * np.linalg.norm(velocity) * line_of_sight_rate
            )
        else:
            commanded_acceleration = np.array([0, 0, 0, 10])

        return commanded_acceleration / np.linalg.norm(
            commanded_acceleration, 1
        )  # normalize


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
