from kspdg.agent_api.base_agent import KSPDGBaseAgent
from kspdg.pe1.e1_envs import PE1_E1_I3_Env
from kspdg.agent_api.runner import AgentEnvRunner
import openai
import json

import os
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

openai.api_key = os.environ.get('OPENAI_API_KEY')

class LLMAgent(KSPDGBaseAgent):
    """An agent that uses ChatGPT to make decisions based on observations."""
    
    def __init__(self, **kwargs):
        super().__init__()

    def get_action(self, observation):
        """ compute agent's action given observation """
        print("get_action called, prompting ChatGPT...")

        user_message = "\n".join([
            f"mission elapsed time: {observation[0]} [s]",
            f"current vehicle (pursuer) mass: {observation[1]} [kg]",
            f"current vehicle (pursuer) propellant (mono prop): {observation[2]} [kg]",
            f"pursuer position x: {observation[3]} [m]",
            f"pursuer position y: {observation[4]} [m]",
            f"pursuer position z: {observation[5]} [m]",
            f"pursuer velocity x: {observation[6]} [m/s]",
            f"pursuer velocity y: {observation[7]} [m/s]",
            f"pursuer velocity z: {observation[8]} [m/s]",
            f"evader position x: {observation[9]} [m]",
            f"evader position y: {observation[10]} [m]",
            f"evader position z: {observation[11]} [m]",
            f"evader velocity x: {observation[12]} [m/s]",
            f"evader velocity y: {observation[13]} [m/s]",
            f"evader velocity z: {observation[14]} [m/s]",
        ])

        messages = [
            {"role": "system", "content": "You are controlling a spacecraft agent in Kerbal Space Program. Your job is to control a pursuit sapcecraft to intercept an evading spacecraft. You will be given observations detailing your position, your speed, the other spacecraft's position, and the other spacecraft's speed. You should reason through how to proceed and call the perform_action function with the throttle values. Only use this function."},
            {"role": "user", "content": user_message},
        ]
        functions = [{
            "name": "perform_action",
            "description": "Send the given throttles to the spacecraft.",
            "parameters": {
                "type": "object",
                "properties": {
                    "forward_throttle": {
                        "type": "integer",
                        "description": "The forward throttle.",
                    },
                    "right_throttle": {
                        "type": "integer",
                        "description": "The right throttle.",
                    },
                    "down_throttle": {
                        "type": "integer",
                        "description": "The down throttle.",
                    },
                },
                "required": ["forward_throttle", "right_throttle", "down_throttle"],
            },
        }]
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            functions=functions,
            function_call="auto",
        )
        response_message = response["choices"][0]["message"]
        print(response_message)
        
        if response_message.get("function_call"):
            duration = 10.0
            available_functions = {
                "perform_action": lambda forward_throttle, right_throttle, down_throttle: [forward_throttle, right_throttle, down_throttle, duration],
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

if __name__ == "__main__":    
    my_agent = LLMAgent()    
    runner = AgentEnvRunner(
        agent=my_agent,
        env_cls=PE1_E1_I3_Env, 
        env_kwargs=None,
        runner_timeout=240,
        debug=False)
    runner.run()

