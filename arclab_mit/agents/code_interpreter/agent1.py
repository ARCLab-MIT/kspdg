from kspdg.agent_api.base_agent import KSPDGBaseAgent
import math
import numpy as np

class Agent(KSPDGBaseAgent):
    def __init__(self):
        super().__init__()
        self.last_log = 0

    def get_action(self, observation):
        pursuer_position = np.array(observation[3:6])
        pursuer_velocity = np.array(observation[6:9])
        evader_position = np.array(observation[9:12])
        evader_velocity = np.array(observation[12:15])

        # Calculate the relative position and velocity
        relative_position = evader_position - pursuer_position
        relative_velocity = evader_velocity - pursuer_velocity

        # Calculate the desired acceleration towards the evader
        desired_acceleration = relative_position - relative_velocity

        # Normalize the desired acceleration
        desired_acceleration_norm = np.linalg.norm(desired_acceleration)
        if desired_acceleration_norm > 0:
            desired_acceleration /= desired_acceleration_norm

        # Apply throttle in the direction of the desired acceleration
        throttle = desired_acceleration

        # logging code: do not modify this
        distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(pursuer_position, evader_position)]))
        if observation[0] - self.last_log >= 5:
            self.last_log = observation[0]
            with open("arclab_mit/agents/code_interpreter/code_interpreter_log.txt", "a") as file:
                file.write(f"{observation[0]},{distance}\n")

        duration = 1
        return {
            "burn_vec": [*throttle, duration],
            "ref_frame": 1
        }