from kspdg.agent_api.base_agent import KSPDGBaseAgent
import math
import numpy as np

class Agent(KSPDGBaseAgent):
    def __init__(self):
        super().__init__()
        self.last_log = 0
        self.prev_distance = None

    def get_action(self, observation):
        pursuer_position = np.array(observation[3:6])
        pursuer_velocity = np.array(observation[6:9])
        evader_position = np.array(observation[9:12])
        evader_velocity = np.array(observation[12:15])

        relative_position = evader_position - pursuer_position
        relative_velocity = evader_velocity - pursuer_velocity

        distance = np.linalg.norm(relative_position)

        if self.prev_distance is None:
            distance_derivative = 0
        else:
            distance_derivative = (distance - self.prev_distance) / (observation[0] - self.last_log)
        self.prev_distance = distance

        cos_angle = np.dot(relative_position, relative_velocity) / (np.linalg.norm(relative_position) * np.linalg.norm(relative_velocity))
        angle_factor = 1 - abs(cos_angle)

        # Estimate time-to-intercept (TTI)
        tti = distance / np.linalg.norm(relative_velocity)

        # Adjust control gains based on distance and TTI
        kp = 0.1 * (distance / 1000) ** 2 + 0.01 * tti
        kd = 0.05 + 0.01 * tti
        kv = 2 * (distance / 1000) + 0.1 * tti

        desired_acceleration = kv * relative_velocity + kp * angle_factor * relative_position / distance - kd * distance_derivative * relative_position / distance

        # Implement bang-bang control strategy
        throttle_magnitude = 1.0 if abs(cos_angle) < 0.8 else 0.0
        throttle = throttle_magnitude * desired_acceleration / np.linalg.norm(desired_acceleration)

        # logging code: do not modify this
        if observation[0] - self.last_log >= 5:
            self.last_log = observation[0]
            with open("arclab_mit/agents/code_interpreter/code_interpreter_log.txt", "a") as file:
                file.write(f"{observation[0]},{distance}\n")

        duration = 1
        return {
            "burn_vec": [*throttle, duration],
            "ref_frame": 1
        }