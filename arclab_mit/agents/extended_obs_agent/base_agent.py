from kspdg.agent_api.base_agent import KSPDGBaseAgent
from kspdg.pe1.e1_envs import PE1_E1_I3_Env
from kspdg.pe1.e3_envs import PE1_E3_I3_Env
from kspdg.agent_api.runner import AgentEnvRunner

class BaseAgent(KSPDGBaseAgent):
    def __init__(self):
        super().__init__()

    def get_action(self, observation):
        return [0,0,0,1]

if __name__ == "__main__":
    runner = AgentEnvRunner(
        agent=BaseAgent(), 
        env_cls=PE1_E1_I3_Env, 
        env_kwargs=None,
        runner_timeout=500,
        debug=False
        )
    runner.run()