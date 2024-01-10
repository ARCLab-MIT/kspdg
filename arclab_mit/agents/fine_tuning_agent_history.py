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
import numpy as np
import sys
import krpc
import datetime
import csv

import random

from kspdg.agent_api.base_agent import KSPDGBaseAgent
from kspdg.agent_api.runner import AgentEnvRunner
from kspdg.pe1.e1_envs import PE1_E1_I3_Env
from kspdg.pe1.e1_envs import PE1_E1_I2_Env
from kspdg.pe1.e1_envs import PE1_E1_I1_Env
from kspdg.pe1.e1_envs import PE1_E1_I4_Env
from kspdg.pe1.e3_envs import PE1_E3_I3_Env
from kspdg.lbg1.lg1_envs import LBG1_LG1_I1_Env
from kspdg.lbg1.lg1_envs import LBG1_LG1_I2_Env
from kspdg.lbg1.lg2_envs import LBG1_LG2_I1_Env
from kspdg.lbg1.lg2_envs import LBG1_LG2_I2_Env
from kspdg.sb1.e1_envs import SB1_E1_I1_Env
from kspdg.sb1.e1_envs import SB1_E1_I2_Env
from kspdg.sb1.e1_envs import SB1_E1_I3_Env
from kspdg.sb1.e1_envs import SB1_E1_I4_Env
from kspdg.sb1.e1_envs import SB1_E1_I5_Env

from os.path import join, dirname
from dotenv import load_dotenv

from sliding_window import SlidingWindow
from arclab_mit.agents.agent_common import State, Action

"""
from arclab_mit.agents.extended_obs_agent.simulate import closest_approach, simulate
from arclab_mit.agents.common import obs_to_state, state_to_message
from astropy import units as u
"""

# Load configuration from .env
# dotenv_path = join(dirname(__file__), 'arclib_mit', 'agents', '.env')
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

# Set OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]

# Load configuration from alex_prompts.txt
# dotenv_path = join(dirname(__file__), 'arclib_mit', 'agents', 'alex_prompts.txt')
dotenv_path = join(dirname(__file__), 'alex_prompts_v2.txt')
load_dotenv(dotenv_path)


class LLMAgent(KSPDGBaseAgent):
    """An agent that uses ChatGPT to make decisions based on observations."""

    def __init__(self, **kwargs):
        super().__init__()
        self.scenario = os.environ['SCENARIO'].lower()
        self.use_relative_coordinates = (os.environ['USE_RELATIVE_COORDINATES'].lower() == "true")
        self.use_short_names = (os.environ['USE_SHORT_ARGUMENT_NAMES'].lower() == "true")
        self.use_enum = (os.environ['USE_ENUM'].lower() == "true")
        if self.use_enum:
            self.functions = [{
                "name": "perform_action",
                "description": "Send the given throttles to the spacecraft.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ft": {
                            "type": "string",
                            "enum": ["backward", "none", "forward"],
                            "description": "The forward throttle.",
                        },
                        "rt": {
                            "type": "string",
                            "enum": ["left", "none", "right"],
                            "description": "The right throttle.",
                        },
                        "dt": {
                            "type": "string",
                            "enum": ["down", "none", "up"],
                            "description": "The down throttle.",
                        },
                    },
                    "required": ["ft", "rt", "dt"],
                },
            }]
        else:
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
                    },
                    "required": ["ft", "rt", "dt"],
                },
            }]

        self.sliding_window_size = int(os.environ["SLIDING_WINDOW_SIZE"])
        self.sliding_window_stride = int(os.environ["SLIDING_WINDOW_STRIDE"])
        self.embed_history = (os.environ['EMBED_HISTORY'].lower() == "true")
        self.skip_all_null_actions = (os.environ['SKIP_ALL_NULL_ACTIONS'].upper() == "true")

        self.closest_distance = sys.float_info.max

        try:
            # Connect to the KRPC server.
            self.conn = krpc.connect()
            # Get the active vessel
            self.vessel = self.conn.space_center.active_vessel
            # Get the celestial body
            self.body = self.vessel.orbit.body
        except Exception as e:
            print ("Exception: " + str(e))
            self.conn = None
            self.vessel = None
            self.body = None

        # Model
        self.model = os.environ['MODEL']
        # scenario
        self.scenario = os.environ['SCENARIO']

        # Create streams to log actions
        if not os.path.exists('logs'):
            self.log = None
            self.log_jsonl = None
        else:
            log_name = "./logs/fine_tune_agent_log_" + self.scenario + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'
            self.log = open(log_name, mode='w', newline='')
            head = ['throttles', 'duration', 'time', 'vehicle_mass', 'vehicle_propellant', 'pursuer_pos_x', 'pursuer_pos_y', 'pursuer_pos_z', 'pursuer_vel_x', 'pursuer_vel_y', 'pursuer_vel_z', 'evader_pos_x', 'evader_pos_y', 'evader_pos_z', 'evader_vel_x', 'evader_vel_y', 'evader_vel_z']
            csv.writer(self.log).writerow(head)

            log_name = log_name.replace("csv", "jsonl")
            self.log_jsonl = open(log_name, mode='w', newline='\n')

        # Interval between actions
        self.duration = 0.5

        # Threshold for prograde alignment
        self.THRESHOLD = 0.1

        # Sliding window parameters
        self.sliding_window_size = int(os.environ["SLIDING_WINDOW_SIZE"])
        self.sliding_window_stride = int(os.environ["SLIDING_WINDOW_STRIDE"])

        # Sliding window
        if self.scenario.startswith('PE'):
            self.sliding_window = SlidingWindow(self.sliding_window_size, 'PE',
                                                self.use_relative_coordinates,
                                                self.use_short_names, self.use_enum,
                                                self.embed_history,
                                                os.environ["PE_SYSTEM_PROMPT"],
                                                os.environ["PE_USER_PROMPT"],
                                                os.environ["PE_CHAIN_OF_THOUGHT"],
                                                os.environ["ASSISTANT_CONTENT"],
                                                "",
                                                "")
        elif self.scenario.startswith('LBG'):
            self.sliding_window = SlidingWindow(self.sliding_window_size, 'LBG',
                                                self.use_relative_coordinates,
                                                self.use_short_names, self.use_enum,
                                                self.embed_history,
                                                os.environ["LBG_SYSTEM_PROMPT"],
                                                os.environ["LBG_USER_PROMPT"],
                                                os.environ["LBG_CHAIN_OF_THOUGHT"],
                                                os.environ["ASSISTANT_CONTENT"],
                                                "",
                                                "")
        elif self.scenario.startswith('SB'):
                self.sliding_window = SlidingWindow(self.sliding_window_size, 'SB',
                                                self.use_relative_coordinates,
                                                self.use_short_names, self.use_enum,
                                                self.embed_history,
                                                os.environ["SB_SYSTEM_PROMPT"],
                                                os.environ["SB_USER_PROMPT"],
                                                os.environ["SB_CHAIN_OF_THOUGHT"],
                                                os.environ["ASSISTANT_CONTENT"],
                                                os.environ["SB_HISTORY_PROMPT"],
                                                os.environ["SB_HISTORY_ITEM_PROMPT"])

    # Return sun position in the celestial body orbital reference frame
    def get_sun_position(self):
        reference_frame = self.body.orbital_reference_frame
        # Get the sun position in the given reference frame
        sun_pos = self.conn.space_center.bodies['Sun'].position(reference_frame)
        return sun_pos

    def get_action(self, observation, sun_position=None):

        # NOTE: Observations are given in the celestial body non-rotating reference frame using right-handed coordinate system.
        # To check it compute vessel position and velocity in the celestial body orbital frame as follows
        #
        # celestial_body_frame = self.body.orbital_reference_frame
        # vessel_frame = self.vessel.reference_frame
        # vessel_velocity_in_celestial_frame = self.conn.space_center.transform_velocity([0,0,0], [0,0,0], vessel_frame, celestial_body_frame)
        # vessel_position_in_celestial_frame = self.conn.space_center.transform_position([0,0,0], vessel_frame, celestial_body_frame)
        #
        # and confirm that these values are close to pursuer position and velocity from observation considering that y-axis has opposite sign
        # due to the fact the kRPC uses left-handed coordinate system. Differences are due to the movement of the vessel reference frame.

        """ compute agent's action given observation """
        print("get_action called, prompting ChatGPT model ..." + self.model)

        # Get the sun position in the given reference frame
        if sun_position is None:
            sun_position = self.get_sun_position()

        # Build state and show it
        state = State(observation, sun_position)
        if state.distance < self.closest_distance:
            self.closest_distance = state.distance
        print(f'Closest distance: {self.closest_distance:.2f}')
        state.show()

        # Add state to sliding window. Action is none since it is unknow at this moment
        self.sliding_window.add(state, None)

        # Create message structure
        messages = self.sliding_window.get_messages()

        action = self.check_response(response=self.get_completion(prompt=messages, model=self.model))
        if action is None:
            _ = self.sliding_window.pop()
            action = [0, 0, 0, 0.1]
        else:
            # Set action in last conversation
            self.sliding_window.set_action(-1, action[0:3])

        """ Log result
         """
        if self.log is not None:
            row = observation
            if not isinstance(row, list):
                row = row.tolist()
            row.insert(0, action[3])
            row.insert(0, action[0:3])
            csv.writer(self.log).writerow(row)
            self.log.flush()

        print("Response action: " + str(action))
        return action

    def clean_response(self, response):
        # clean function args
        function_args = response

        # Eliminate "=> "
        function_args = function_args.replace('=>', '')

        # Find and extract perform action arguments
        index = function_args.find("perform_action(")
        if index != -1:
            # Extract perform_action arguments
            function_args = function_args[index + len("perform_action("):]
            index = function_args.find(")")
            if index != -1:
                function_args = function_args[:index]
            # Surround arguments with quotes
            function_args = function_args.replace("ft", '"ft"')
            function_args = function_args.replace("rt", '"rt"')
            function_args = function_args.replace("dt", '"dt"')
            function_args = function_args.replace(',\n}', '\n}')
        function_args = function_args.replace("]", '}')
        function_args = function_args.replace("[", '{')

        # Now function arguments should be of the form:
        #   "{\n  \"ft\": 1,\n  \"rt\": -1,\n  \"dt\": 1\n}"
        #   "ft: 1, rt: -1, dt: 1"
        #   "ft -1 rt 1 dt 0"
        #   "-1, 0, 0"
        # Transform to the first form
        index = function_args.find("ft")
        if index == -1:
            # Case
            #   "1, 0 ,0"
            action = function_args.split(',')
            function_args = "{\n  \"ft\": " + action[0] + ",\n  \"rt\": " + action[1] + ",\n  \"dt\": " + action[2] + "\n}"
        else:
            index = function_args.find("{")
            if index == -1:
                # Cases:
                #   "ft: 1, rt: -1, dt: 1"
                #   "ft -1 rt 1 dt 0"

                # Add "{" and "}"
                function_args = "{" + function_args + "}"
                # Add colons
                index = function_args.find(":")
                if index == -1:
                    colons = ":"
                else:
                    colons = ""
                index = function_args.find(",")
                if index == -1:
                    comma = ","
                else:
                    comma = ""

                # Surround argument names with quotes
                function_args = function_args.replace("ft", '"ft"' + colons)
                function_args = function_args.replace("rt", comma + '"rt"' + colons)
                function_args = function_args.replace("dt", comma + '"dt"' + colons)

        return function_args

    def check_response(self, response):
        if response is None:
            return None
        print("Response message: " + str(response))
        if response.get("function_call"):
            duration = self.duration
            available_functions = {
                "perform_action": lambda ft, rt, dt: [ft, rt, dt, duration],
            }
            function_name = response["function_call"]["name"]
            if function_name not in available_functions:
                print("error: LLM called wrong function, name:", function_name)
                function_response = [0, 0, 0, 0.1]
            else:
                function_to_call = available_functions[function_name]

                # Get function arguments
                function_args = response["function_call"]["arguments"]
                try:
                    function_args = self.clean_response(function_args)
                    function_args = json.loads(function_args)
                    function_response = function_to_call(**function_args)
                    if self.use_enum:
                        function_response = Action.from_enum(function_response)
                except Exception as ex:
                    print(function_args)
                    function_response = [0, 0, 0, 0.1]
        else:
            print("error: LLM did not call function")
            function_response = [0, 0, 0, 0.1]

        return function_response

    def get_completion(self, prompt, model="gpt-4-1106-preview"):

        print("Prompt:")
        for message in prompt:
            print(message)

        # Perform chat completion
        time_before = time.time()
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=prompt,
                functions=self.functions,
                max_tokens = 100, # limit output tokens (enough for valid responses)
                temperature=0  # randomness, cool approach if we want to adjust some param with this
            )
            status = "success"
            status_message = None
            result = response.choices[0].message
            token_usage = response["usage"].to_dict()
        except Exception as e:
            print ("Exception: " + str(e))
            status = "error"
            status_message = str(e)
            response = None
            result = None
            token_usage = {}
        time_after = time.time()

        """ Log conversation
        """
        if self.log_jsonl is not None:
            log_entry = {
                "name": self.scenario,
                "kind": "llm",
                "status_code": status,
                "status_message": status_message,
                "metadata": {
                    "temperature": 0,
                    "token_usage": token_usage,
                    "model_name": model,
                },
                "start_time_ms": time_before * 1000,
                "end_time_ms": time_after * 1000,
                "inputs": prompt,
                "outputs": response,
            }
            json.dump(log_entry, self.log_jsonl)
            self.log_jsonl.write('\n')
            self.log_jsonl.flush()

        print("Chat completion took " + str(time_after - time_before) + " seconds")
        self.first_completion = False

        return result


if __name__ == "__main__":

    scenario = os.environ['SCENARIO']

    scenarios = dict()
    scenarios["PE1_E1_I1"] = PE1_E1_I1_Env
    scenarios["PE1_E1_I2"] = PE1_E1_I2_Env
    scenarios["PE1_E1_I3"] = PE1_E1_I3_Env
    scenarios["PE1_E1_I4"] = PE1_E1_I4_Env
    scenarios["PE1_E3_I3"] = PE1_E3_I3_Env

    scenarios["LBG1_LG1_I1"] = LBG1_LG1_I1_Env
    scenarios["LBG1_LG1_I2"] = LBG1_LG1_I2_Env
    scenarios["LBG1_LG2_I1"] = LBG1_LG2_I1_Env
    scenarios["LBG1_LG2_I2"] = LBG1_LG2_I2_Env

    scenarios["SB1_E1_I1"] = SB1_E1_I1_Env
    scenarios["SB1_E1_I2"] = SB1_E1_I2_Env
    scenarios["SB1_E1_I3"] = SB1_E1_I3_Env
    scenarios["SB1_E1_I4"] = SB1_E1_I4_Env
    scenarios["SB1_E1_I5"] = SB1_E1_I5_Env

    if scenario not in scenarios:
        print("Invalid scenario: " + scenario + " not in " + str(scenarios.keys()))
        sys.exit(1)

    print("Running scenario: " + scenario)
    print("Model: " + os.environ['MODEL'])

    my_agent = LLMAgent()
    runner = AgentEnvRunner(
        agent=my_agent,
        env_cls=scenarios[scenario],
        env_kwargs=None,
        runner_timeout=240,
        debug=False)
    runner.run()
