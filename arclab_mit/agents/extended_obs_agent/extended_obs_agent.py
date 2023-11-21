"""
Run this as a MODULE: `python -m arclab_mit.agents.extended_obs_agent.extended_obs_agent`
or else imports won't work
"""
from kspdg.agent_api.base_agent import KSPDGBaseAgent
from kspdg.pe1.e1_envs import PE1_E1_I3_Env
from kspdg.agent_api.runner import AgentEnvRunner
import openai
import json
import time
import numpy as np
from arclab_mit.agents.extended_obs_agent.simulate import simulate
from arclab_mit.agents.common import round_tuple, obs_to_state, state_to_message

import os
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '..', '..', '..', '.env')
print(dotenv_path)
load_dotenv(dotenv_path)

openai.api_key = os.environ.get('OPENAI_API_KEY')
print(openai.api_key)

class LLMAgent(KSPDGBaseAgent):
    def __init__(self):
        super().__init__()
        self.prev = (None, "no action")
        self.directions_prompt = ""

    def get_action(self, observation):
        print("get_action called, prompting ChatGPT...")
        state = obs_to_state(observation)

        if self.prev[1] in ["left", "right", "up", "down"]:
            forward_dir, right_dir, up_dir = self.calculate_directions(self.prev[0], observation, self.prev[1])
            self.directions_prompt = "\n".join([
                "Here are the estimated directions of each movement:",
                f"Forward (x,y,z): {round_tuple(forward_dir,3)}",
                f"Right (x,y,z): {round_tuple(right_dir,3)}",
                f"Up (x,y,z): {round_tuple(up_dir,3)}",
            ])
        
        self.prev = (observation, "no action")

        user_message = "\n".join([
            "Here is the current state:",
            state_to_message(state),
            "Here is what the state will be in 10 seconds, if you don't provide any thrust:",
            state_to_message(simulate(state, 10)),
            "Here is what the state will be in 30 seconds, if you don't provide any thrust:",
            state_to_message(simulate(state, 30)),
            self.directions_prompt,
            "Based on this information, determine the optimal direction to apply thrust to intercept the evading spacecraft. You will be measured on your closest approach distance, not the time taken, so you should take your time and be careful not to overshoot the evading spacecraft by going too fast. Instead, make sure to use left/right and up/down thrusters to intercept it correctly. Use the perform_action function to adjust the spacecraft's trajectory. Call this function directly. Do not run Python or any other programming language.",
        ])
        print(user_message)

        system_prompt_path = "arclab_mit/jason_system_prompt.txt"
        with open(system_prompt_path, 'r') as file:
            system_prompt = file.read()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        # define the callable functions. there is just 1 function, perform_action, which takes 3 parameters
        # we will set duration = 10s for now, and chatgpt has no control over this
        functions = [{
            "name": "perform_action",
            "description": "Perform a navigation action by specifying a direction for the spacecraft.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action_direction": {
                        "type": "string",
                        "enum": ["forward", "back", "left", "right", "up", "down", "no action"],
                        "description": "The direction in which to perform the action."
                    },
                },
                "required": ["action_direction"],
            },
        }]
        
        start_time = time.time()
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            functions=functions,
            function_call="auto",
        )
        print("Time taken (seconds): ", time.time() - start_time)
        response_message = response["choices"][0]["message"]
        print(response_message)
        
        # getting results of the function call and returning the response
        if response_message.get("function_call"):
            duration = 1.0
            def perform_action_func(action_direction):
                self.prev = (observation, action_direction)
                return {
                    "forward": [1.0, 0.0, 0.0],
                    "back": [-1.0, 0.0, 0.0],
                    "left": [0.0, -1.0, 0.0],
                    "right": [0.0, 1.0, 0.0],
                    "up": [0.0, 0.0, -1.0],
                    "down": [0.0, 0.0, 1.0],
                    "no action": [0.0, 0.0, 0.0]
                }[action_direction] + [duration]

            available_functions = {
                # the actual "function"; it just takes the parameters and returns the action vector
                "perform_action": perform_action_func,
            }
            function_name = response_message["function_call"]["name"]
            if function_name not in available_functions:
                print("error: LLM called wrong function, name:", function_name)
                return [0,0,0,0.1]

            function_to_call = available_functions[function_name]
            function_args = json.loads(response_message["function_call"]["arguments"])
            function_response = function_to_call(**function_args)
            return function_response

        print("error: LLM did not call function")
        return [0,0,0,0.1]
    
    def calculate_directions(self, obs1, obs2, action):
        """
        take 2 observations and an action that is either "left", "right", "up", "down"
        calculate the orientation of all directions during the first observation
        """
        assert action in ["left", "right", "up", "down"]
        time1 = obs1[0]
        time2 = obs2[0]
        state1 = obs_to_state(obs1)
        state2 = obs_to_state(obs2)
        pred_state2 = simulate(state1, time2 - time1)
        vel_diff = np.array(state2[1]) - np.array(pred_state2[1])
        direction = vel_diff / np.linalg.norm(vel_diff)
        pos_diff = np.array(state2[0]) - np.array(pred_state2[0])
        forward_dir = pos_diff / np.linalg.norm(pos_diff)

        # TODO: some of the signs might be wrong (from cross product), verify them
        if action == "right":
            right_dir = direction
        if action == "left":
            right_dir = -direction
        if action == "up":
            right_dir = -np.cross(forward_dir, direction)
        if action == "down":
            right_dir = np.cross(forward_dir, direction)
        right_dir /= np.linalg.norm(right_dir)
        up_dir = np.cross(forward_dir, right_dir)
        up_dir /= np.linalg.norm(up_dir)

        return tuple(forward_dir), tuple(right_dir), tuple(up_dir)

my_agent = LLMAgent()    
runner = AgentEnvRunner(
    agent=my_agent,
    env_cls=PE1_E1_I3_Env, 
    env_kwargs=None,
    runner_timeout=240,
    debug=False)
runner.run()