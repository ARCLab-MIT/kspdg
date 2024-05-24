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
        if observation[0] - self.last_log >= 5:
            self.last_log = observation[0]
            with open("arclab_mit/agents/code_interpreter/code_interpreter_log.txt", "a") as file:
                file.write(f"{observation[0]},{distance}\n")

        # Normalize the relative position to get the direction vector
        norm = math.sqrt(sum([x**2 for x in relative_position]))
        direction_vector = [x / norm for x in relative_position]

        duration = 1

        return {
            "burn_vec": [direction_vector[0], direction_vector[1], direction_vector[2], duration],
            "ref_frame": 1  # celestial reference frame
        }
