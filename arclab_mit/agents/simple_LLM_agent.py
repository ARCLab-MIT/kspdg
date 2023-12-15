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
import numpy
import sys

import random

from krpc.services import krpc

from kspdg.agent_api.base_agent import KSPDGBaseAgent
from kspdg.agent_api.runner import AgentEnvRunner
from kspdg.pe1.e1_envs import PE1_E1_I3_Env
from kspdg.pe1.e1_envs import PE1_E1_I2_Env
from kspdg.pe1.e1_envs import PE1_E1_I1_Env
from kspdg.pe1.e1_envs import PE1_E1_I4_Env

from os.path import join, dirname
from dotenv import load_dotenv

from kspdg.sb1.e1_envs import SB1_E1_I1_Env, SB1_E1_I2_Env, SB1_E1_I3_Env, SB1_E1_I4_Env, SB1_E1_I5_Env

# Load configuration from .env
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

# Set OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]

# Load configuration from alex_prompts.txt
dotenv_path = join(dirname(__file__), 'alex_prompts.txt')
load_dotenv(dotenv_path)

class SimpleLLMAgent(KSPDGBaseAgent):
    """An agent that uses ChatGPT to make decisions based on observations."""

    def __init__(self, **kwargs):
        super().__init__()
        self.functions = [{
            "name": "perform_action",
            "description": "Send the given throttles to the spacecraft.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ft": {
                        "type": "integer",
#                        "multipleOf" : 0.01,
                        "minimum": -1,
                        "maximum": 1,
                        "description": "The forward throttle.",
                    },
                    "rt": {
                        "type": "integer",
#                        "multipleOf" : 0.01,
                        "minimum": -1,
                        "maximum": 1,
                        "description": "The right throttle.",
                    },
                    "dt": {
                        "type": "integer",
#                        "multipleOf" : 0.01,
                        "minimum": -1,
                        "maximum": 1,
                        "description": "The down throttle.",
                    },
                },
                "required": ["ft", "rt", "dt"],
            },
        }]
        self.first_completion = True
        self.closest_distance = sys.float_info.max
        self.use_relative_coordinates = (os.environ['USE_RELATIVE_COORDINATES'].lower() == "true")
        self.use_short_argument_names = (os.environ['USE_SHORT_ARGUMENT_NAMES'].lower() == "true")

        # Connect to the KRPC server.
        self.conn = vessel = krpc.connect()
        # Get the active vessel
        self.vessel = self.conn.space_center.active_vessel
        # Get the body
        self.body = self.vessel.orbit.body

        # Model
        self.model = os.environ['MODEL']
        # scenario
        self.scenario = os.environ['SCENARIO']
        # Ignore time
        self.ignore_time = os.environ['IGNORE_TIME']


    def evaluation (self, observation):
        position = [observation[9] - observation[3], observation[10] - observation[4], observation[11] - observation[5]]
        deltav = [observation[12] - observation[6], observation[13] - observation[7], observation[14] - observation[8]]
        mission_time = observation[0]
        fuel = observation[2]
        distance = numpy.linalg.norm(position, ord=2)
        velocity = numpy.linalg.norm(deltav, ord=2)
        if (distance < self.closest_distance):
            self.closest_distance = distance
        print(f'Closest Distance: {self.closest_distance:.2f}')
        print(f'Distance: {distance:.2f}')
        print(f'Velocity: {velocity:.2f}')
        print(f'Mission time: {mission_time:.2f}')
        print(f'Fuel: {fuel:.2f}')
        return


    def get_action(self, observation):
        """ compute agent's action given observation """
        print("get_action called, prompting ChatGPT...")

        persuader_position = [observation[3], observation[4], observation[5]]
        persuader_velocity = [observation[6], observation[7], observation[8]]
        evader_position = [observation[9], observation[10], observation[11]]
        evader_velocity = [observation[12], observation[13], observation[14]]

        message = os.environ['USER_PROMPT'] + " {"
        if self.ignore_time == "False":
            if self.use_short_argument_names:
                message += \
                    str(f"\"t\": {observation[0]}, ")
            else:
                message += \
                    str(f"\"time\": {observation[0]}, ")

        if self.use_relative_coordinates:
            if self.use_short_argument_names:
                message += \
                    str(f"\"m\": {observation[1]}, ") + \
                    str(f"\"f\": {observation[2]}, ") + \
                    str(f"\"rx\": {evader_position[0] - persuader_position[0]}, ") + \
                    str(f"\"ry\": {evader_position[1] - persuader_position[1]}, ") + \
                    str(f"\"rz\": {evader_position[2] - persuader_position[2]}, ") + \
                    str(f"\"rvx\": {evader_velocity[0] - persuader_velocity[0]}, ") + \
                    str(f"\"rvy\": {evader_velocity[1] - persuader_velocity[1]}, ") + \
                    str(f"\"rvz\": {evader_velocity[2] - persuader_velocity[2]}") + \
                    "}"
            else:
                message += \
                    str(f"\"vehicle_mass\": {observation[1]}, ") + \
                    str(f"\"vehicle_propellant\": {observation[2]}, ") + \
                    str(f"\"relative_pos_x\": {evader_position[0] - persuader_position[0]}, ") + \
                    str(f"\"relative_pos_y\": {evader_position[1] - persuader_position[1]}, ") + \
                    str(f"\"relative_pos_z\": {evader_position[2] - persuader_position[2]}, ") + \
                    str(f"\"relative_vel_x\": {evader_velocity[0] - persuader_velocity[0]}, ") + \
                    str(f"\"relative_vel_y\": {evader_velocity[1] - persuader_velocity[1]}, ") + \
                    str(f"\"relative_vel_z\": {evader_velocity[2] - persuader_velocity[2]}") + \
                    "}"
        else:
            if self.use_short_argument_names:
                message += \
                    str(f"\"m\": {observation[1]}, ") + \
                    str(f"\"f\": {observation[2]}, ") + \
                    str(f"\"px\": {persuader_position[0]}, ") + \
                    str(f"\"py\": {persuader_position[1]}, ") + \
                    str(f"\"pz\": {persuader_position[2]}, ") + \
                    str(f"\"pvx\": {persuader_velocity[0]}, ") + \
                    str(f"\"pvy\": {persuader_velocity[1]}, ") + \
                    str(f"\"pvz\": {persuader_velocity[2]}, ") + \
                    str(f"\"ex\": {evader_position[0]}, ") + \
                    str(f"\"ey\": {evader_position[1]}, ") +  \
                    str(f"\"ez\": {evader_position[2]}, ") + \
                    str(f"\"evx\": {evader_velocity[0]}, ") + \
                    str(f"\"evy\": {evader_velocity[1]}, ") + \
                    str(f"\"evz\": {evader_velocity[2]}") + \
                    "}"
            else:
                message += \
                    str(f"\"vehicle_mass\": {observation[1]}, ") + \
                    str(f"\"vehicle_propellant\": {observation[2]}, ") + \
                    str(f"\"persuader_pos_x\":  {persuader_position[0]}, ") + \
                    str(f"\"persuader_pos_y\":  {persuader_position[1]}, ") + \
                    str(f"\"persuader_pos_z\":  {persuader_position[2]}, ") + \
                    str(f"\"persuader_vel_x\":  {persuader_velocity[0]}, ") + \
                    str(f"\"persuader_vel_y\":  {persuader_velocity[1]}, ") + \
                    str(f"\"persuader_vel_z\":  {persuader_velocity[2]}, ") + \
                    str(f"\"evader_pos_x\":  {evader_position[0]}, ") + \
                    str(f"\"evader_pos_y\":  {evader_position[1]}, ") +  \
                    str(f"\"evader_pos_z\":  {evader_position[2]}, ") + \
                    str(f"\"evader_vel_x\":  {evader_velocity[0]}, ") + \
                    str(f"\"evader_vel_y\":  {evader_velocity[1]}, ") + \
                    str(f"\"evader_vel_z\":  {evader_velocity[2]}") + \
                    "}"

        self.evaluation (observation)
        try:
            action = self.check_response(response=self.get_completion(prompt=message))
        except Exception as e:
            print ("Exception: " + str(e))
            action = [0, 0, 0, 0.1]

        print(action)
        return {
            "burn_vec": action,
            # throttle in x-axis, throttle in y-axis, throttle in z-axis, duration [s]
            "ref_frame": 0  # burn_vec expressed in agent vessel's right-handed body frame.
            # i.e. forward throttle, right throttle, down throttle,
            # Can also use rhcbci (1) and rhntw (2) ref frames
        }

    def check_response(self, response):
        print(response)
        if response.get("function_call"):
            duration = 0.5
            available_functions = {
                "perform_action": lambda ft, rt, dt: [ft, rt, dt, duration],
            }
            function_name = response["function_call"]["name"]
            if function_name not in available_functions:
                print("error: LLM called wrong function, name:", function_name)
                return [0, 0, 0, 0.1]

            function_to_call = available_functions[function_name]

            # Get and clean function args
            function_args = response["function_call"]["arguments"]
            index = function_args.find("perform_action(")
            if index != -1:
                # Extract perform_action arguments
                function_args = function_args[index+len("perform_action("):]
                index = function_args.find(")")
                if index != -1:
                    function_args = function_args[:index]
                # Surround arguments with quotes
                function_args = function_args.replace("ft", '"ft"')
                function_args = function_args.replace("rt", '"rt"')
                function_args = function_args.replace("dt", '"dt"')
                function_args = function_args.replace(',\n}', '\n}')
            function_args = json.loads(function_args)

            # Get response
            function_response = function_to_call(**function_args)
            return function_response

        print("error: LLM did not call function")
        return [0,0,0,0.1]

    def get_completion(self, prompt, model="ft:gpt-3.5-turbo-1106:personal::8MFqjElw"):
        model = "gpt-3.5-turbo-1106"
        if self.first_completion:
            messages = [{"role": "system", "content": "You are a language model calculator that has to calculate the spacecraft's throttles\
                                                       You aim to solve a pursuer evader problem, where you are given the pursuer and evader's position and velocity as well as other parameters.\
                                                       After reasoning, please call the perform_action function giving ###numerical arguments only.###. Show throttle calculations."}]
        else:
            messages = []
        messages.append({"role": "user", "content": prompt})
        time_before = time.time()
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            functions=self.functions,
#            max_tokens = 100, # limit output tokens (enough for valid responses)
            temperature=0  # randomness, cool approach if we want to adjust some param with this
        )
        time_after = time.time()
        print ("Completion took " + str(time_after - time_before) + " seconds")
        self.first_completion = False

        return response.choices[0].message


if __name__ == "__main__":

    scenario = os.environ['SCENARIO']

    scenarios = dict()
    scenarios["PE1_E1_I1"] = PE1_E1_I1_Env
    scenarios["PE1_E1_I2"] = PE1_E1_I2_Env
    scenarios["PE1_E1_I3"] = PE1_E1_I3_Env
    scenarios["PE1_E1_I4"] = PE1_E1_I4_Env
    scenarios["SB1_E1_I1"] = SB1_E1_I1_Env
    scenarios["SB1_E1_I2"] = SB1_E1_I2_Env
    scenarios["SB1_E1_I3"] = SB1_E1_I3_Env
    scenarios["SB1_E1_I4"] = SB1_E1_I4_Env
    scenarios["SB1_E1_I5"] = SB1_E1_I5_Env

    if not scenario in scenarios:
        print("Invalid scenario: " + scenario + " not in " + str(scenarios.keys()))
        sys.exit(1)

    my_agent = SimpleLLMAgent()
    runner = AgentEnvRunner(
        agent=my_agent,
        env_cls=scenarios[scenario],
        env_kwargs=None,
        runner_timeout=240,
        debug=False)
    runner.run()
