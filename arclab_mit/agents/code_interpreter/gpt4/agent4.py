from kspdg.agent_api.base_agent import KSPDGBaseAgent
import math

class Agent(KSPDGBaseAgent):
    def __init__(self):
        super().__init__()
        self.last_log = 0
        self.Kp = 0.5  # Proportional gain for Proportional Navigation
        self.Kd = 0.1  # Derivative gain for velocity correction
        self.last_distance = None  # Track last distance for rate of change calculation

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
        closure_rate = (self.last_distance - distance) if self.last_distance else 0
        self.last_distance = distance

        # Log distance every 5 seconds
        if observation[0] - self.last_log >= 5:
            self.last_log = observation[0]
            with open("arclab_mit/agents/code_interpreter/code_interpreter_log.txt", "a") as file:
                file.write(f"{observation[0]},{distance}\n")

        # Normalize vectors with checks for zero division
        def normalize(v):
            norm = math.sqrt(sum(x**2 for x in v))
            return [x / norm for x in v] if norm else v

        direction_vector = normalize(relative_position)
        relative_velocity_normalized = normalize(relative_velocity)

        # Proportional Navigation Logic: Closing velocity vector correction
        correction_vector = [self.Kp * (relative_position[i] - self.Kd * relative_velocity[i]) for i in range(3)]
        correction_vector_normalized = normalize(correction_vector)

        # Combining direction vector with velocity correction
        final_vector = [direction_vector[i] + self.Kd * correction_vector_normalized[i] for i in range(3)]
        final_vector_normalized = normalize(final_vector)

        # Dynamic throttle control based on distance and closure rate
        throttle = min(1.0, max(0.1, 1 - (distance / 5000) + 0.5 * closure_rate / distance))  # Adjust as necessary

        duration = 1  # Duration of the burn in seconds

        return {
            "burn_vec": [final_vector_normalized[0] * throttle, final_vector_normalized[1] * throttle, final_vector_normalized[2] * throttle, duration],
            "ref_frame": 1  # celestial reference frame
        }

