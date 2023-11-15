from kspdg.agent_api.base_agent import KSPDGBaseAgent
from kspdg.pe1.e1_envs import PE1_E1_I3_Env
from kspdg.pe1.e3_envs import PE1_E3_I3_Env
from kspdg.agent_api.runner import AgentEnvRunner
from simulate import simulate

class CalibrationAgent(KSPDGBaseAgent):
    """goes in each direction to figure out what directions everything is"""
    def __init__(self):
        super().__init__()
        self.counter = -1

    def get_action(self, observation):
        print("called get_action, time:", observation[0])
        self.counter += 1

        state = observation[3:6], observation[6:9], observation[9:12], observation[12:15]
        if self.counter == 0:
            self.prev_state = state
            self.prev_time = observation[0]
            return [0,1,0,0.2]
        elif self.counter >= 1:
            curr_time = observation[0]
            print(curr_time - self.prev_time)
            pred_state = simulate(self.prev_state, curr_time - self.prev_time)
            print(tuple(a-b for a,b in zip(state[1], pred_state[1])))

            self.prev_state = state
            self.prev_time = observation[0]
        

if __name__ == "__main__":
    runner = AgentEnvRunner(
        agent=CalibrationAgent(),
        env_cls=PE1_E1_I3_Env,
        env_kwargs=None,
        runner_timeout=500,
        debug=False
        )
    runner.run()
