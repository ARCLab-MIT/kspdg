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

import csv
import datetime
import json
import os
import sys
import time

import krpc
import numpy as np
import openai
from dotenv import load_dotenv

from arclab_mit.agents.agent_common import State, Action, setup_scenarios, set_env_paths
from kspdg.agent_api.base_agent import KSPDGBaseAgent
from kspdg.agent_api.runner import AgentEnvRunner
from kspdg.lbg1.lbg1_base import LadyBanditGuardGroup1Env
from kspdg.pe1.e1_envs import PE1_E1_I1_Env
from kspdg.pe1.e1_envs import PE1_E1_I3_Env
from kspdg.pe1.e1_envs import PE1_E1_I4_Env
from kspdg.pe1.e3_envs import PE1_E3_I3_Env
from kspdg.pe1.e3_envs import PE1_E3_I4_Env
from kspdg.pe1.e2_envs import PE1_E2_I3_Env
from kspdg.pe1.pe1_base import PursuitEvadeGroup1Env
from kspdg.sb1.e1_envs import SB1_E1_I1_Env
from kspdg.sb1.e1_envs import SB1_E1_I2_Env
from kspdg.sb1.e1_envs import SB1_E1_I3_Env
from kspdg.sb1.e1_envs import SB1_E1_I4_Env
from kspdg.sb1.e1_envs import SB1_E1_I5_Env
from kspdg.sb1.sb1_base import SunBlockingGroup1Env

from arclab_mit.agents.sliding_window import SlidingWindow

import httpx
import asyncio
from pydantic import BaseModel

set_env_paths()

"""
from arclab_mit.agents.extended_obs_agent.simulate import closest_approach, simulate
from arclab_mit.agents.common import obs_to_state, state_to_message
from astropy import units as u
"""


class LlamaAPIClient:
    def __init__(self, base_url: str, timeout=60):
        self.base_url = base_url
        self.timeout = timeout

    def get(self, endpoint: str, params: dict = None):
        response = httpx.get(self.base_url + endpoint, params=params, timeout=self.timeout)
        response.raise_for_status()  # Raises an error for bad responses
        return response.json()

    def post(self, endpoint: str, data: dict = None, json: dict = None):
        response = httpx.post(self.base_url + endpoint, data=data, json=json, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def put(self, endpoint: str, data: dict = None, json: dict = None):
        response = httpx.put(self.base_url + endpoint, data=data, json=json, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def delete(self, endpoint: str):
        response = httpx.delete(self.base_url + endpoint)
        response.raise_for_status()
        return response.json()

    def close(self):
        httpx.aclose()


class LlamaAgent(KSPDGBaseAgent):
    """An agent that uses Llama 3 to make decisions based on observations."""

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
        self.use_prograde = (os.environ['USE_PROGRADE'].lower() == "true")
        self.use_cot = (os.environ['USE_COT'].lower() == "true")
        self.use_cot_speed_limit = (os.environ['USE_COT_SPEED_LIMIT'].lower() == "true")

        self.sliding_window_size = int(os.environ["SLIDING_WINDOW_SIZE"])
        self.sliding_window_stride = int(os.environ["SLIDING_WINDOW_STRIDE"])
        self.embed_history = (os.environ['EMBED_HISTORY'].lower() == "true")
        self.skip_all_null_actions = (os.environ['SKIP_ALL_NULL_ACTIONS'].upper() == "true")

        self.closest_distance = sys.float_info.max
        self.weighted_score = sys.float_info.max
        self.init_mass = None

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
            log_name = "./logs/llama_fine_tune_agent_log_" + self.scenario + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'
            self.log = open(log_name, mode='w', newline='')
            if self.scenario.lower().startswith("lbg"):
                if self.use_prograde:
                    head = ['throttles', 'duration', 'time', 'vehicle_mass', 'vehicle_propellant', 'pursuer_pos_x',
                            'pursuer_pos_y', 'pursuer_pos_z', 'pursuer_vel_x', 'pursuer_vel_y', 'pursuer_vel_z',
                            'evader_pos_x', 'evader_pos_y', 'evader_pos_z', 'evader_vel_x', 'evader_vel_y',
                            'evader_vel_z', 'guard_pos_x', 'guard_pos_y', 'guard_pos_z', 'guard_vel_x', 'guard_vel_y',
                            'guard_vel_z', 'vessel_up', 'prograde', 'weighted_score']
                else:
                    head = ['throttles', 'duration', 'time', 'vehicle_mass', 'vehicle_propellant', 'pursuer_pos_x',
                            'pursuer_pos_y', 'pursuer_pos_z', 'pursuer_vel_x', 'pursuer_vel_y', 'pursuer_vel_z',
                            'evader_pos_x', 'evader_pos_y', 'evader_pos_z', 'evader_vel_x', 'evader_vel_y',
                            'evader_vel_z', 'guard_pos_x', 'guard_pos_y', 'guard_pos_z', 'guard_vel_x', 'guard_vel_y',
                            'guard_vel_z', 'weighted_score']
            else:
                if self.use_prograde:
                    head = ['throttles', 'duration', 'time', 'vehicle_mass', 'vehicle_propellant', 'pursuer_pos_x',
                            'pursuer_pos_y', 'pursuer_pos_z', 'pursuer_vel_x', 'pursuer_vel_y', 'pursuer_vel_z',
                            'evader_pos_x', 'evader_pos_y', 'evader_pos_z', 'evader_vel_x', 'evader_vel_y',
                            'evader_vel_z', 'vessel_up', 'prograde', 'weighted_score']
                else:
                    head = ['throttles', 'duration', 'time', 'vehicle_mass', 'vehicle_propellant', 'pursuer_pos_x',
                            'pursuer_pos_y', 'pursuer_pos_z', 'pursuer_vel_x', 'pursuer_vel_y', 'pursuer_vel_z',
                            'evader_pos_x', 'evader_pos_y', 'evader_pos_z', 'evader_vel_x', 'evader_vel_y',
                            'evader_vel_z', 'weighted_score']

            csv.writer(self.log).writerow(head)

            log_name = log_name.replace("csv", "jsonl")
            self.log_jsonl = open(log_name, mode='w', newline='\n')

        # Interval between actions
        self.duration = 1

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
                                                self.use_prograde,
                                                self.use_cot,
                                                self.use_cot_speed_limit,
                                                self.embed_history,
                                                os.environ["PE_SYSTEM_PROMPT"],
                                                os.environ["PE_USER_PROMPT"],
                                                os.environ["PE_CHAIN_OF_THOUGHT"],
                                                os.environ["ASSISTANT_CONTENT"],
                                                "",
                                                "")
        else:
            self.sliding_window = SlidingWindow(self.sliding_window_size, 'SB',
                                                self.use_relative_coordinates,
                                                self.use_short_names, self.use_enum,
                                                self.use_prograde,
                                                self.use_cot,
                                                self.use_cot_speed_limit,
                                                self.embed_history,
                                                os.environ["SB_SYSTEM_PROMPT"],
                                                os.environ["SB_USER_PROMPT"],
                                                os.environ["SB_CHAIN_OF_THOUGHT"],
                                                os.environ["ASSISTANT_CONTENT"],
                                                os.environ["SB_HISTORY_PROMPT"],
                                                os.environ["SB_HISTORY_ITEM_PROMPT"])

        self.pe1Env = PursuitEvadeGroup1Env("pe1_i1_init")
        self.lbg1Env = LadyBanditGuardGroup1Env("lbg1_i1_init")
        self.sb1Env = SunBlockingGroup1Env("sb1_i1_init")

        # FastAPI client
        base_url = os.environ['LLAMA_URL']
        self.client = LlamaAPIClient(base_url)

    # Return sun position in the celestial body orbital reference frame
    def get_sun_position(self):
        reference_frame = self.body.orbital_reference_frame
        # Get the sun position in the given reference frame
        sun_pos = self.conn.space_center.bodies['Sun'].position(reference_frame)
        return sun_pos

    def get_action(self, observation, sun_position=None):

        # NOTE: Observations are given in the celestial body celestial reference frame using right-handed coordinate system.
        # See https://krpc.github.io/krpc/tutorials/reference-frames.html#tutorial-reference-frames for details about krpc reference frames.
        # Vessel's reference frame in krpc is left-handed:
        #
        #   x is pointing to the right
        #   y axis is pointing forward
        #   z axis is pointing down
        #
        # To check it compute vessel position and velocity in the celestial body orbital reference frame as follows
        #
        #   celestial_body_frame = self.body.orbital_reference_frame
        #   vessel_frame = self.vessel.reference_frame
        #   vessel_orbital_frame = self.vessel.orbital_reference_frame
        #
        #   vessel_velocity_in_celestial_frame = self.conn.space_center.transform_velocity([0,0,0], [0,0,0], vessel_frame, celestial_body_frame)
        #   vessel_position_in_celestial_frame = self.conn.space_center.transform_position([0,0,0], vessel_frame, celestial_body_frame)
        #
        # and confirm that these values are close to pursuer position and velocity from observation considering that y-axis has opposite sign
        # due to the fact the kRPC uses left-handed coordinate system. Differences are due to the movement of the vessel reference frame.
        #
        # The vessel velocity direction in the celestial body non-rotating reference frame, assuming non-accelerating vessel, is given by:
        #
        #   self.conn.space_center.transform_direction([0, 1, 0], vessel_orbital_frame, celestial_body_frame)
        #
        # Vessel's nose pointing direction is obtained as follows:
        #
        #   pointing_direction = np.array(self.conn.space_center.transform_direction([0, 1, 0], vessel_frame, celestial_body_frame))
        #   pointing_direction[1] = -pointing_direction[1]
        #
        # This direction should be the same as the direction of the evader's position relative to pursuer's since vessel is pointing at evader.
        #
        # y-axis (forward) in vessel's reference frame is pointing at target and x-axis to the right. Thus
        # direction [0, 1, 0] is transformed to rel_position direction in celestial body reference frame:
        #
        # self.conn.space_center.transform_direction([0, 1, 0], vessel_frame, celestial_body_frame)
        #
        # x-axis in vessel's reference frame is transformed to:
        # x_axis = self.conn.space_center.transform_direction([1, 0, 0], vessel_frame, celestial_body_frame).
        #
        # Prograde marker in navball is given by relative velocity vector

        """ compute agent's action given observation """
        print("\nget_action called, prompting ChatGPT model ..." + self.model)

        # Get vessel up direction in celestial body reference frame
        vessel_up = None
        vessel_up = self.conn.space_center.transform_direction((0, 0, 1),
                                                               self.vessel.reference_frame,
                                                               self.body.orbital_reference_frame)
        # BE CAREFUL
        vessel_up = State.lh_to_rh(vessel_up)

        # Get the sun position in the given reference frame
        if sun_position is None:
            sun_position = self.get_sun_position()

        # Build state and show it
        state = State(observation, vessel_up, sun_position)

        if state.distance < self.closest_distance:
            self.closest_distance = state.distance
            if self.init_mass is None:
                self.init_mass = state.vehicle_mass
            weighted_score = 0.0
            if self.scenario.lower().startswith("pe"):
                weighted_score = self.pe1Env.get_weighted_score(state.distance, state.velocity, state.mission_time, self.init_mass - state.vehicle_mass)
            elif self.scenario.lower().startswith("lbg"):
                lb_dist = state.distance
                lg_dist = np.linalg.norm(state.evader_position - state.guard_position, ord=2)
                weighted_score = self.lbg1Env.get_weighted_score(lb_dist, lg_dist)
            elif self.scenario.lower().startswith("sb"):
                """ TODO: Implement weighted score for SB1  """
            self.weighted_score = weighted_score
        print(f'Closest distance: {self.closest_distance:.2f}')
        print(f'Weighted score: {self.weighted_score:.2f}')

        state.show()

        # Add state to sliding window. Action is none since it is unknow at this moment
        self.sliding_window.add(state, None)

        # Create message structure
        messages = self.sliding_window.get_messages()

        print("Model: " + self.model)
        action = self.check_response(response=self.get_completion(prompt=messages, model=self.model))
        # action[2] = -action[2]
        if action is None:
            _ = self.sliding_window.pop()
            action = [0, 0, 0, 0.1]
        else:
            # Set action in last conversation
            self.sliding_window.set_action(-1, action[0:3])

            # Check if prograde direction is aligned with action
            #
            # NOTE: Navball prograde is in fact the retrograde direction
            #
            prograde_dir = state.get_retrograde()
            prograde_dir = prograde_dir / np.linalg.norm(prograde_dir, ord=2)
            ft = 1
            rt = 1 if prograde_dir[0] > 0 else -1
            dt = 1 if prograde_dir[2] > 0 else -1
            if state.distance < 1500:
                if state.velocity > 20:
                    ft = -1
                else:
                    ft = 0
            if rt != action[1] or dt != action[2]:
                print(f"Warning: Action does not align prograde={prograde_dir}. Recommended rotations: {rt} and {dt}")
            approaching = prograde_dir[1] < 0
            if state.approaching != approaching:
                print(f"Warning: Approaching is {state.approaching} but prograde is {prograde_dir}")

        """ Log result
        """
        if self.log is not None:
            row = observation[0:15]
            row.insert(0, action[3])
            row.insert(0, action[0:3])
            if self.use_prograde:
                vessel_up = state.vessel_up
                row.append(list(vessel_up))
                prograde = state.get_prograde()
                row.append(prograde.tolist())
            row.append(self.weighted_score)
            print("Row: " + str(row))
            csv.writer(self.log).writerow(row)
            self.log.flush()

        print("Response action: " + str(action))
        return action

    def clean_response(self, response):
        # clean function args
        function_args = response

        # Eliminate "=> "
        function_args = function_args.replace('=>', '')

        # Find and extract perform action arguments.
        # Rather than searching for "perform_action(" we search for the first argument name
        # index = function_args.find("perform_action(")
        index = function_args.find('{"ft":')
        if index != -1:
            # Extract perform_action arguments
            # function_args = function_args[index + len("perform_action("):]
            function_args = function_args[index:]
            # index = function_args.find(")")
            index = function_args.find("}")
            if index != -1:
                function_args = function_args[:index+1]
            # Surround arguments with quotes
            function_args = function_args.replace("ft:", '"ft":')
            function_args = function_args.replace("rt:", '"rt":')
            function_args = function_args.replace("dt:", '"dt":')
            function_args = function_args.replace(',\n}', '\n}')
        function_args = function_args.replace("]", '}')
        function_args = function_args.replace("[", '{')

        index = function_args.find("Backward throttle(")
        if index != -1:
            print("Found Backward throttle")

        # Replace argument names used by CoT prompt
        #
        function_args = function_args.replace("Forward throttle", "ft")
        function_args = function_args.replace("Backward throttle", "ft")
        function_args = function_args.replace("Right throttle", "rt")
        function_args = function_args.replace("Up throttle", "dt")
        function_args = function_args.replace("Down throttle", "dt")

        # CoT prompt with speed limit returns the following ft values:
        # - full
        # - 0.5
        # We map them to "forward"
        #
        function_args = function_args.replace("full", "forward")
#        function_args = function_args.replace("0.5", "forward")

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

        if function_args[0] == "down":
            function_args[0] = "forward"

        return function_args

    def check_response(self, response):
        if response is None:
            return None

        print("Response message: " + str(response))
        duration = self.duration
        available_functions = {
            "perform_action": lambda ft, rt, dt: [ft, rt, dt, duration],
        }
        if response.get("function_call"):
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
            function_name = "perform_action"
            function_to_call = available_functions[function_name]
            try:
                content = response["content"]
                # Assume that the response contains the perform_action call
                # index = content.find(function_name + "(")
                index = 0
                if index != -1:
                    # Get function args from content
                    function_args = self.clean_response(content)
                    function_args = json.loads(function_args)

                    print ("Function args: " + str(function_args))
                    # Add missing arguments
                    if "ft" not in function_args.keys():
                        function_args["ft"] = 0
                    if "rt" not in function_args.keys():
                        function_args["rt"] = 0
                    if "dt" not in function_args.keys():
                        function_args["dt"] = 0

                    # Eliminate extra arguments
                    for key in function_args.keys():
                        if key not in ["ft", "rt", "dt"]:
                            del function_args[key]

                    function_response = function_to_call(**function_args)
                    if self.use_enum:
                        if type(function_response[0]) == str:
                            function_response = Action.from_enum(function_response)
                else:
                    print("error: Response did not include call to perform_action")
                    function_response = [0, 0, 0, 0.1]
            except Exception as ex:
                print("Exception processing function_args:" + str(function_args))
                function_response = [0, 0, 0, 0.1]

        return function_response

    def get_completion(self, prompt, model="gpt-4-1106-preview"):

        print("Prompt:")
        for message in prompt:
            print(message)

        # Perform chat completion
        time_before = time.time()
        try:
            system_prompt = ""
            user_prompt = ""
            for message in prompt:
                if message["role"] == "system":
                    system_prompt = message["content"]
                elif message["role"] == "user":
                    user_prompt = message["content"]

            response = self.client.post("/generate/", json={"system_prompt": system_prompt, "user_msg": user_prompt, "model_answer": ""})

            status = "success"
            status_message = None
            result = response
            token_usage = {}
            """
            if self.use_cot:
                # Cannot use function calling with CoT
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=prompt,
                    functions=self.functions,
                    temperature=0  # randomness, cool approach if we want to adjust some param with this
                )
            else:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=prompt,
                    functions=self.functions,
                    max_tokens = 150, # limit output tokens (enough for valid responses)
                    temperature=0  # randomness, cool approach if we want to adjust some param with this
                )
            status = "success"
            status_message = None
            result = response.choices[0].message
            token_usage = response["usage"].to_dict()
            """
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

    scenarios = setup_scenarios()

    if scenario not in scenarios:
        print("Invalid scenario: " + scenario + " not in " + str(scenarios.keys()))
        sys.exit(1)

    print("Running scenario: " + scenario)
    print("Model: " + os.environ['MODEL'])

    my_agent = LlamaAgent()
    runner = AgentEnvRunner(
        agent=my_agent,
        env_cls=scenarios[scenario],
        env_kwargs=None,
        runner_timeout=240,
        debug=False)
    runner.run()
