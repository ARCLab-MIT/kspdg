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
import krpc

import random

from kspdg.agent_api.base_agent import KSPDGBaseAgent
from kspdg.agent_api.runner import AgentEnvRunner
from kspdg.pe1.e1_envs import PE1_E1_I3_Env
from kspdg.pe1.e1_envs import PE1_E1_I2_Env
from kspdg.pe1.e1_envs import PE1_E1_I1_Env
from kspdg.pe1.e1_envs import PE1_E1_I4_Env

from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

openai.api_key = os.environ["OPENAI_API_KEY"]
openai.api_key = "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"
openai.api_key ="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" # Alex's key

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
                    "ft": {
                        "type": "integer",
                        "minimum": -1,
                        "maximum": 1,
                        "description": "The forward throttle.",
                    },
                    "rt": {
                        "type": "integer",
                        "minimum": -1,
                        "maximum": 1,
                        "description": "The right throttle.",
                    },
                    "dt": {
                        "type": "integer",
                        "minimum": -1,
                        "maximum": 1,
                        "description": "The down throttle.",
                    },
                    "nft": {
                        "type": "integer",
                        "minimum": -1,
                        "maximum": 1,
                        "description": "Next forward throttle.",
                    },
                    "nrt": {
                        "type": "integer",
                        "minimum": -1,
                        "maximum": 1,
                        "description": "Next right throttle.",
                    },
                    "ndt": {
                        "type": "integer",
                        "minimum": -1,
                        "maximum": 1,
                        "description": "Next down throttle.",
                    },
                },
                "required": ["ft", "rt", "dt", "nft", "nrt", "ndt"],
            },
        }]

        self.first_completion = True
        self.closest_distance = sys.float_info.max

        # Sliding Window

        # Connect to the KRPC server.
        self.conn = vessel = krpc.connect()
        # Get the active vessel
        self.vessel = self.conn.space_center.active_vessel
        # Get the body
        self.body = self.vessel.orbit.body

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

        # NOTE: Observations are given in the celestial body non-rotating reference frame using right-handed coordinate system.
        # To check it compute vessel position and velocity in the celestial body orbital frame as follows
        #
        # celestial_body_frame = self.body.orbital_reference_frame
        # vessel_frame = self.vessel.reference_frame
        # vessel_velocity_in_celestial_frame = self.conn.space_center.transform_velocity([0,0,0], [0,0,0], vessel_frame, celestial_body_frame)
        # vessel_position_in_celestial_frame = self.conn.space_center.transform_position([0,0,0], vessel_frame, celestial_body_frame)
        #
        # and confirm that these values are close to persuader position and velocity from observation considering that y-axis has opposite sign
        # due to the fact the kRPC uses left-handed coordinate system. Differences are due to the movement of the vessel reference frame.

        """ compute agent's action given observation """
        print("get_action called, prompting ChatGPT...")

        persuader_position = [observation[3], observation[4], observation[5]]
        persuader_velocity = [observation[6], observation[7], observation[8]]
        evader_position = [observation[9], observation[10], observation[11]]
        evader_velocity = [observation[12], observation[13], observation[14]]

        message = "Best action for\n".join([
            f"t: {observation[0]} [s]",
            f"m: {observation[1]} [kg]",
            f"f: {observation[2]} [kg]",
            f"px: {persuader_position[0]} [m]",
            f"py: {persuader_position[1]} [m]",
            f"pz: {persuader_position[2]} [m]",
            f"pvx: {persuader_velocity[0]} [m/s]",
            f"pvy: {persuader_velocity[1]} [m/s]",
            f"pvz: {persuader_velocity[2]} [m/s]",
            f"ex: {evader_position[0]} [m]",
            f"ey: {evader_position[1]} [m]",
            f"ez: {evader_position[2]} [m]",
            f"evx: {evader_velocity[0]} [m/s]",
            f"evy: {evader_velocity[1]} [m/s]",
            f"evz: {evader_velocity[2]} [m/s]",
        ])
        self.evaluation (observation)
        try:
            action = self.check_response(response=self.get_completion(prompt=message))
        except Exception as e:
            print ("Exception: " + str(e))
            action = [0,0,0,0.1]

        print(action)
        return action

    def check_response(self, response):
        print(response)
        if response.get("function_call"):
            duration = 0.5
            available_functions = {
                "perform_action": lambda ft, rt, dt, nft, nrt, ndt: [ft, rt, dt, nft, nrt, ndt, duration],
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
                function_args = function_args.replace("nft", '"nft"')
                function_args = function_args.replace("nrt", '"nrt"')
                function_args = function_args.replace("ndt", '"ndt"')
                function_args = function_args.replace(',\n}', '\n}')
            function_args = function_args.replace("]", '}')
            function_args = function_args.replace("[", '{')
            index = function_args.find("\"ft")
            if index != -1:
                index = function_args.find("\"ft", index+1)
                if index != -1:
                    # Eliminate second occurrences of arguments
                    function_args = function_args[:index-1] + "}"

            try:
                function_args = json.loads(function_args)
            except Exception as ex:
                print (function_args)
                return [0,0,0,0.1]

            # Get response
            function_response = function_to_call(**function_args)
            # Eliminate nft, nrt, ndt from function_response
            function_response[3:6] = []
            return function_response

        print("error: LLM did not call function")
        return [0,0,0,0.1]

    def get_completion(self, prompt, model="ft:gpt-3.5-turbo-1106:personal::8MFqjElw"):
        #model = "ft:gpt-3.5-turbo-1106:personal:kspgpt:8NMUS0kq" # First model
        #model = "gpt-3.5-turbo-1106"
        #model = "ft:gpt-3.5-turbo-1106:personal:kspgpt:8Nd7OJkC" # Second model
        #model = "ft:gpt-3.5-turbo-1106:personal::8OpiTR5F" # Fine tuning with all scenarios and 0.5s samples
        #model = "ft:gpt-3.5-turbo-1106:personal::8OsW2it4" # Fine tuning with E3 scenario and 0.5s samples
        model = "ft:gpt-3.5-turbo-1106:personal::8OtbAUxq" # Fine tuning with E2 scenario and without [0,0,0] actions
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

    if len(sys.argv) < 2:
        sys.exit(1)
    scenario = sys.argv[1]

    scenarios = {}
    scenarios["PE1_E1_I1"] = PE1_E1_I1_Env
    scenarios["PE1_E1_I2"] = PE1_E1_I2_Env
    scenarios["PE1_E1_I3"] = PE1_E1_I3_Env
    scenarios["PE1_E1_I4"] = PE1_E1_I4_Env
    if not scenario in scenarios:
        print("Invalid scenario: " + scenario + " not in " + str(scenarios.keys()))
        sys.exit(1)

    my_agent = LLMAgent()
    runner = AgentEnvRunner(
        agent=my_agent,
        env_cls=scenarios[scenario],
        env_kwargs=None,
        runner_timeout=240,
        debug=False)
    runner.run()
