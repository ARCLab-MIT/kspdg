from kspdg.agent_api.base_agent import KSPDGBaseAgent
from kspdg.pe1.e1_envs import PE1_E1_I3_Env
from kspdg.agent_api.runner import AgentEnvRunner
from extended_obs_agent.extended_obs_agent import ExtendedObsAgent
import json
import os

output_json_path = os.path.join(os.path.dirname(__file__), 'extended_obs_agent_log.json')

class LoggingAgent(KSPDGBaseAgent):
    """
    wrapper around another agent, that logs observation/actions
    """
    def __init__(self, agent):
        super().__init__()
        self.agent = agent

    def get_action(self, observation):
        action = self.agent.get_action(observation)
        with open(output_json_path, 'a') as file:
            json.dump({
                "observation": observation,
                "action": action
            }, file)
            file.write("\n")
        
        return action

if __name__ == "__main__":
    runner = AgentEnvRunner(
        agent=LoggingAgent(ExtendedObsAgent()), 
        env_cls=PE1_E1_I3_Env, 
        env_kwargs=None,
        runner_timeout=500,
        debug=False
        )
    runner.run()