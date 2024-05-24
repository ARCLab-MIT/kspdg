from kspdg.agent_api.base_agent import KSPDGBaseAgent
import math

class Agent(KSPDGBaseAgent):
    def __init__(self):
        super().__init__()
        self.last_log = 0
        self.Kp = 0.5  # Proportional gain for Proportional Navigation
        self.Kd = 0.1  # Derivative gain for velocity correction

    def get_action(self, observation):
        # Extract relevant data from observation
        pursuer_position = observation[3:6]
        pursuer_velocity = observation[6:9]
        evader_position = observation[9:12]
        evader_velocity = observation[12:15]

        # Calculate relative position and velocity
        relative_position = [evader_position[i] - pursuer_position[i] for i in range(3)]
        relative_velocity = [evader_velocity[i] - pursuer_velocity[i] for i in range(3)]

        # Calculate the distance to the evader
        distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(pursuer_position, evader_position)]))

        # Log distance every 5 seconds
        if observation[0] - self.last_log >= 5:
            self.last_log = observation[0]
            with open("arclab_mit/agents/code_interpreter/code_interpreter_log.txt", "a") as file:
                file.write(f"{observation[0]},{distance}\n")

        # Normalizing vectors
        norm_position = math.sqrt(sum([x**2 for x in relative_position]))
        direction_vector = [x / norm_position for x in relative_position]

        norm_velocity = math.sqrt(sum([v**2 for v in relative_velocity]))
        relative_velocity_normalized = [v / norm_velocity for v in relative_velocity]

        # Proportional Navigation Logic: Closing velocity vector correction
        correction_vector = [self.Kp * (relative_position[i] * norm_velocity - pursuer_velocity[i] * norm_velocity) for i in range(3)]
        norm_correction = math.sqrt(sum([x**2 for x in correction_vector]))
        correction_vector_normalized = [x / norm_correction for x in correction_vector]

        # Combining direction vector with velocity correction
        final_vector = [direction_vector[i] + self.Kd * correction_vector_normalized[i] for i in range(3)]
        norm_final = math.sqrt(sum([x**2 for x in final_vector]))
        final_vector_normalized = [x / norm_final for x in final_vector]

        # Dynamic throttle control
        throttle = min(1.0, 0.1 + 0.9 * (5000 / max(5000, distance)))  # Adjust as necessary

        duration = 1  # Duration of the burn in seconds

        return {
            "burn_vec": [final_vector_normalized[0] * throttle, final_vector_normalized[1] * throttle, final_vector_normalized[2] * throttle, duration],
            "ref_frame": 1  # celestial reference frame
        }
