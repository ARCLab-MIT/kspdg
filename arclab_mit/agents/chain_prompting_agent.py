# Copyright (c) 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

"""
This script is a "Hello World" for writing agents that can interact with
a KSPDG environment.

Instructions to Run:
- Start KSP game application.
- Select Start Game > Play Missions > Community Created > pe1_i3 > Continue
- In kRPC dialog box click Add server. Select Show advanced settings and select Auto-accept new clients. Then select Start Server
- In a terminal, run this script

"""
import os
import openai
import json

import time
import random

from kspdg.agent_api.base_agent import KSPDGBaseAgent
from kspdg.agent_api.runner import AgentEnvRunner
from kspdg.pe1.e1_envs import PE1_E1_I3_Env

from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

import JSON_parsing
import time

openai.api_key = os.environ["OPENAI_API_KEY"]
print ("OpenAI API key: " + openai.api_key)



class LLMAgent(KSPDGBaseAgent):
    """An agent that uses ChatGPT to make decisions based on observations."""

    def __init__(self, **kwargs):
        super().__init__()
        self.functions = [{
            "name": "perform_action",
            "description": "Send the given throttles to the spacecraft.",
            "parameters": {
                "type": "object",
                "properties": {
                    "forward_throttle": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "The forward throttle.",
                    },
                    "right_throttle": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "The right throttle.",
                    },
                    "down_throttle": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "The down throttle.",
                    },
                },
                "required": ["forward_throttle", "right_throttle", "down_throttle"],
            },
        }]
        self.first_completion = True

    def get_action(self, observation):
        """ compute agent's action given observation """
        print("get_action called, prompting ChatGPT...")

        inicio = time.time()

        message = "\n".join([
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
        action = self.check_response(response=self.get_completion(prompt=message))
        print(action)

        fin = time.time()
        print("Tiempo de ejecución: ", fin - inicio)

        return action

    def check_response(self, response):
        print(response)
        if response.get("function_call"):
            duration = 1.0
            available_functions = {
                "perform_action": lambda forward_throttle, right_throttle, down_throttle: [forward_throttle, right_throttle, down_throttle, duration],
            }
            function_name = response["function_call"]["name"]
            if function_name not in available_functions:
                print("error: LLM called wrong function, name:", function_name)
                return [0, 0, 0, 0.1]

            function_to_call = available_functions[function_name]
            function_args = json.loads(response["function_call"]["arguments"])
            function_response = function_to_call(**function_args)
            return function_response

        print("error: LLM did not call function")
        return [0,0,0,0.1]

    def get_completion(self, prompt, model="gpt-4-1106-preview"):
        if self.first_completion:
            messages = [{"role": "system", "content": "You are a language model calculator that has to calculate the spacecraft's throttles\
                                                       You aim to solve a pursuer evader problem, where you are given the pursuer and evader's position and velocity as well as other parameters.\
                                                       After reasoning, please call the perform_action function giving ###numerical arguments only.###"}]
        else:
            messages = []
        history = JSON_parsing.get_history("pe1_i3")
        """
                    for i in range(0,10):
            messages.append(history[random.randint(0, len(history) - 1)])
        
        """
        for list in history:
            messages.append(list)
        messages.append({"role": "user", "content": prompt})
        time_before = time.time()
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            functions=self.functions,
            temperature=0  # randomness, cool approach if we want to adjust some param with this
        )
        time_after = time.time()
        print ("Completion took " + str(time_after - time_before) + " seconds")
        self.first_completion = False

        return response.choices[0].message


if __name__ == "__main__":
    my_agent = LLMAgent()
    runner = AgentEnvRunner(
        agent=my_agent,
        env_cls=PE1_E1_I3_Env,
        env_kwargs=None,
        runner_timeout=240,
        debug=False)
    runner.run()
