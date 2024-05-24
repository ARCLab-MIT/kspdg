from kspdg.agent_api.base_agent import KSPDGBaseAgent
import math

class Agent(KSPDGBaseAgent):
    def __init__(self):
        super().__init__()
        self.last_log = 0

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

        # Normalize the relative position to get the direction vector for the burn
        norm_position = math.sqrt(sum([x**2 for x in relative_position]))
        direction_vector = [x / norm_position for x in relative_position]

        # Adjust throttle based on distance, smaller throttle when further away
        throttle = min(1.0, 0.1 + 0.9 * (5000 / max(5000, distance)))  # Example throttle control

        # Combine relative velocity adjustment into the direction vector
        norm_velocity = math.sqrt(sum([v**2 for v in relative_velocity]))
        if norm_velocity > 0:
            relative_velocity = [v / norm_velocity for v in relative_velocity]
            direction_vector = [direction_vector[i] + relative_velocity[i] for i in range(3)]
            norm_direction = math.sqrt(sum([x**2 for x in direction_vector]))
            direction_vector = [x / norm_direction for x in direction_vector]

        duration = 1  # Duration of the burn in seconds

        return {
            "burn_vec": [direction_vector[0] * throttle, direction_vector[1] * throttle, direction_vector[2] * throttle, duration],
            "ref_frame": 1  # celestial reference frame
        }
