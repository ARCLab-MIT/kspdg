from kspdg.agent_api.base_agent import KSPDGBaseAgent
from kspdg.pe1.e1_envs import PE1_E1_I3_Env
from kspdg.agent_api.runner import AgentEnvRunner
import pandas as pd
import argparse

class OpenInterpreterAgent(KSPDGBaseAgent):
    """An agent that uses open interpreter to plan actions"""
    def __init__(self, system_prompt_fp=None, **kwargs):
        """
        Args:
            system_prompt_fp : str
                filepath to system prompt file
        """
        super().__init__()
        if system_prompt_fp is not None:
            with open(system_prompt_fp, 'r') as f:
                self.system_prompt = f.read()
                print("SYSTEM PROMPT: \n", self.system_prompt)
        self.observations = []
    
    def get_action(self, observation):
        """ compute agent's action given observation
        """

        self.observations.append(observation)
        # print the observation to align with the headers
        print(observation)
        
        return [1.0, 0, 0, 5.0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Open Interpreter Agent.')
    parser.add_argument('--system_prompt_fp', type=str, required=False,
                        help='System Prompt Filepath')
    args = parser.parse_args()
    
    my_agent = OpenInterpreterAgent(args.system_prompt_fp)    
    runner = AgentEnvRunner(
        agent=my_agent, 
        env_cls=PE1_E1_I3_Env, 
        env_kwargs=None,
        runner_timeout=240,     # agent runner that will timeout after 100 seconds
        debug=False)
    runner.run()

