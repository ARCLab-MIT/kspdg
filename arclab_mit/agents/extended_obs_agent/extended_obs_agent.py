from kspdg.agent_api.base_agent import KSPDGBaseAgent
from kspdg.pe1.e1_envs import PE1_E1_I3_Env
from kspdg.pe1.e3_envs import PE1_E3_I3_Env
from kspdg.agent_api.runner import AgentEnvRunner
import openai
import json
import time
import numpy as np

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from arclab_mit.agents.extended_obs_agent.simulate import simulate, closest_approach
from arclab_mit.agents.common import obs_to_state, state_to_message, round_arr

from dotenv import load_dotenv
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env')
load_dotenv(dotenv_path)
openai.api_key = os.environ.get('OPEN_API_KEY')

class ExtendedObsAgent(KSPDGBaseAgent):
    def __init__(self):
        super().__init__()
        system_prompt_path = os.path.join(os.path.dirname(__file__), "system_prompt.md")
        with open(system_prompt_path, 'r') as file:
            system_prompt = file.read()
        self.messages = [{"role": "system", "content": system_prompt}]
        self.use_manual_response = True
    
    def get_manual_response(self, state):
        state = [np.array(ele) for ele in state]
        acc_dir = round_arr((state[2] - state[0])/np.linalg.norm(state[0] - state[2]), 1)
        return {
            "role": "assistant",
            "content": f"Because of the large distance between us and the evader, we should accelerate towards the evader to close the distance. We should accelerate in the opposite signs of the relative positions to close the distance. Now, and at the closest approach, the accelerate direction should be {acc_dir}.",
            "function_call": {
                "name": "apply_throttle",
                "arguments": "{\n  \"throttle\": [" + ", ".join(map(str, acc_dir)) + "]\n}"
            }
        }

    def get_action(self, observation):
        print("=" * 100)
        print("=" * 100)
        print("get_action called, prompting ChatGPT...")
        state = obs_to_state(observation)

        closest_state, closest_time = closest_approach(state, 300)
        user_message = "\n".join([
            f"Elapsed Time: {observation[0]} [s]",
            "Current state:",
            state_to_message(state),
            # "Here is what the state will be in 40 seconds, if you don't provide any thrust:",
            # state_to_message(simulate(state, 40)),
            f"If your spacecraft and the evader both stop accelerating, this will be the simulated closest approach (happens at time {closest_time}s):",
            state_to_message(closest_state),
        ])
        self.messages.append({"role": "user", "content": user_message})
        print(user_message)
        print("=" * 30)
        
        # repeatedly pop the first pair (index 1 and 2) to not exceed context length
        while True:
            total_len_estimate = 0 # estimate becuase idk how function calls count toward context
            for message in self.messages:
                if message["content"] is not None:
                    total_len_estimate += len(message["content"])
            if total_len_estimate <= 15000: # leaving some buffer for function calling tokens, so not full context length
                break
            self.messages = [self.messages[0]] + self.messages[3:]

        # print(self.messages)

        if self.use_manual_response:
            response_message = self.get_manual_response(state)
            self.use_manual_response = False
        else:
            start_time = time.time()
            response = openai.ChatCompletion.create(
                # model="gpt-3.5-turbo",
                model="gpt-3.5-turbo-16k",
                messages=self.messages,
                functions=[{
                    "name": "apply_throttle",
                    "description": "Move the pursuer spacecraft with a (x,y,z) throttle.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "throttle": {
                                "type": "array",
                                "items": {
                                    "type": "number",
                                    "minimum": -1.0,
                                    "maximum": 1.0
                                },
                                "description": "An array of three floats, each between -1.0 and 1.0, that specify the (x,y,z) throttle values."
                            }
                        },
                        "required": ["throttle"],
                    },
                }],
                function_call="auto",
            )
            print("Time taken (seconds): ", time.time() - start_time)
            response_message = response["choices"][0]["message"]
        
        print(response_message)

        self.messages.append(response_message)
        
        # getting results of the function call and returning the response
        if response_message.get("function_call"):
            duration = 5.0
            def get_action(throttle):
                return {
                    "burn_vec": throttle + [duration],
                    "ref_frame": 1 # celestial ref frame
                }
            available_functions = {
                "apply_throttle": get_action,
            }
            function_name = response_message["function_call"]["name"]
            if function_name not in available_functions:
                print("error: LLM called wrong function, name:", function_name)
                self.messages.append({"role": "user", "content": "Error: you called the wrong function. Please only use apply_throttle, not python"})
                return [0,0,0,0.1]
            function_to_call = available_functions[function_name]

            try:
                function_args = json.loads(response_message["function_call"]["arguments"])
                function_response = function_to_call(**function_args)
                return function_response
            except:
                print("error occured while parsing arguments")
                return [0,0,0,0.1]

        print("error: LLM did not call function")
        self.messages.append({"role": "user", "content": "Error: you did not call a function. Remember to use apply_throttle at the end of every response."})
        return [0,0,0,0.1]

if __name__ == "__main__":
    my_agent = ExtendedObsAgent()    
    runner = AgentEnvRunner(
        agent=my_agent,
        env_cls=PE1_E3_I3_Env, 
        env_kwargs=None,
        runner_timeout=120,
        debug=False)
    runner.run()