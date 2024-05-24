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

        # Calculate the relative position and velocity
        relative_position = evader_position - pursuer_position
        relative_velocity = evader_velocity - pursuer_velocity

        # Calculate the distance between the pursuer and the evader
        distance = np.linalg.norm(relative_position)

        # Calculate the rate of change of distance
        if self.prev_distance is None:
            distance_derivative = 0
        else:
            distance_derivative = (distance - self.prev_distance) / (observation[0] - self.last_log)
        self.prev_distance = distance

        # Calculate the angle between relative position and relative velocity
        cos_angle = np.dot(relative_position, relative_velocity) / (np.linalg.norm(relative_position) * np.linalg.norm(relative_velocity))
        angle_factor = 1 - abs(cos_angle)

        # Calculate the desired acceleration towards the evader
        kp = 0.1 * (distance / 1000) ** 2  # Proportional control term
        kd = 0.05  # Derivative control term
        kv = 2 * (distance / 1000)  # Velocity control term
        desired_acceleration = kv * relative_velocity + kp * angle_factor * relative_position / distance - kd * distance_derivative * relative_position / distance

        # Normalize the desired acceleration
        desired_acceleration_norm = np.linalg.norm(desired_acceleration)
        if desired_acceleration_norm > 0:
            desired_acceleration /= desired_acceleration_norm

        # Apply throttle in the direction of the desired acceleration
        throttle = desired_acceleration

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