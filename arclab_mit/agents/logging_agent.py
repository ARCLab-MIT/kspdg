from kspdg.agent_api.base_agent import KSPDGBaseAgent
from kspdg.pe1.e1_envs import PE1_E1_I3_Env
from kspdg.agent_api.runner import AgentEnvRunner
import json
import os

output_json_path = os.path.join(os.path.dirname(__file__), 'outputs.json')

class LoggingAgent(KSPDGBaseAgent):
    """
    agent that does nothing, but prints observations (for debugging use)
    """
    def __init__(self):
        super().__init__()

    def get_action(self, observation):
        print(observation)
        return [0,0,0,5]

if __name__ == "__main__":
    for _ in range(10):
        runner = AgentEnvRunner(
            agent=LoggingAgent(), 
            env_cls=PE1_E1_I3_Env, 
            env_kwargs=None,
            runner_timeout=500,
            debug=False
            )
        runner.run()