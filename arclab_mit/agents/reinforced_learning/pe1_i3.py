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
import random
from gymnasium import ObservationWrapper
import openai
import json
import numpy as np

from kspdg.agent_api.base_agent import KSPDGBaseAgent
from kspdg.agent_api.runner import AgentEnvRunner
from kspdg.pe1.e1_envs import PE1_E1_I3_Env

from os.path import join, dirname
from dotenv import load_dotenv
from scipy.spatial import distance


dotenv_path = join(dirname(__file__), ".env")
load_dotenv(dotenv_path)

openai.api_key = os.environ["OPENAI_API_KEY"]


class LLMAgent(KSPDGBaseAgent):
    """An agent that uses ChatGPT to make decisions based on observations."""

    def __init__(self, **kwargs):
        super().__init__()
        self.functions = [
            {
                "name": "perform_action",
                "description": "Send the given throttles to the spacecraft.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "forward_throttle": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "The forward throttle.",
                        },
                        "right_throttle": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "The right throttle.",
                        },
                        "down_throttle": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "The down throttle.",
                        },
                    },
                    "required": ["forward_throttle", "right_throttle", "down_throttle"],
                },
            }
        ]

    def get_action(self, observation):
        """compute agent's action given observation"""
        print("get_action called, prompting ChatGPT...")

        # Enhance agent's ability to analyze distance
        pursuer_position = observation[3:6]
        evader_position = observation[9:12]
        dist = distance.euclidean(pursuer_position, evader_position)
        remaining_distance = "remaining distance to target: " + str(dist) + " [m]"

        message = "\n".join(
            [
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
            ]
        )

        action = self.check_response(
            response=self.get_completion(prompt=message), observation=observation
        )
        print(action)
        return action

    def check_response(self, response, observation):
        """Check the response from the model, call appropriate function if necessary"""
        if response.get("function_call"):
            duration = 10.0
            available_functions = {
                "perform_action": lambda forward_throttle, right_throttle, down_throttle: [
                    forward_throttle,
                    right_throttle,
                    down_throttle,
                    duration,
                ],
            }

            function_name = response["function_call"]["name"]
            if function_name not in available_functions:
                print("error: LLM called wrong function, name:", function_name)
                # Calculate direction vector to evader
                pursuer_position = np.array(observation[3:6])
                evader_position = np.array(observation[9:12])
                direction_vector = evader_position - pursuer_position
                direction_vector = direction_vector / np.linalg.norm(direction_vector)
                return list(np.append(direction_vector, 0.1))

            function_to_call = available_functions[function_name]
            function_args = json.loads(response["function_call"]["arguments"])
            function_response = function_to_call(**function_args)
            return function_response

        print("error: LLM did not call function")
        return [random.random(), random.random(), random.random(), 0.1]

    def get_completion(self, prompt, model="gpt-3.5-turbo"):
        messages = [
            {"role": "user", "content": prompt},
            {
                "role": "system",
                "content": "You are a language model calculator that has to calculate the spacecraft's throttles\
                                                   You aim to solve a pursuer evader problem, where you are given the pursuer and evader's position and velocity as well as other parameters.\
                                                   After reasoning, please call the perform_action function giving numerical arguments only.",
            },
        ]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            functions=self.functions,
            temperature=0,  # randomness, cool approach if we want to adjust some param with this
        )
        return response.choices[0].message


if __name__ == "__main__":
    my_agent = LLMAgent()
    runner = AgentEnvRunner(
        agent=my_agent,
        env_cls=PE1_E1_I3_Env,
        env_kwargs=None,
        runner_timeout=240,
        debug=False,
    )
    runner.run()
