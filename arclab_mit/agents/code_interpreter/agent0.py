from kspdg.agent_api.base_agent import KSPDGBaseAgent
import math

class Agent(KSPDGBaseAgent):
    def __init__(self):
        super().__init__()
        self.last_log = 0

    def get_action(self, observation):
        pursuer_position = observation[3:6]
        pursuer_velocity = observation[6:9]
        evader_position = observation[9:12]
        evader_velocity = observation[12:15]

        # logging code: do not modify this
        distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(pursuer_position, evader_position)]))
        if observation[0] - self.last_log >= 5:
            self.last_log = observation[0]
            with open("arclab_mit/agents/code_interpreter/code_interpreter_log.txt", "a") as file:
                file.write(f"{observation[0]},{distance}\n")

        duration = 1

        return {
            "burn_vec": [0, 0, 0, duration],
            "ref_frame": 1
        }
