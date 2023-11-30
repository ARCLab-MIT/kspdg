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
import datetime
import csv

import random

from kspdg.agent_api.base_agent import KSPDGBaseAgent
from kspdg.agent_api.runner import AgentEnvRunner
from kspdg.pe1.e1_envs import PE1_E1_I3_Env
from kspdg.pe1.e1_envs import PE1_E1_I2_Env
from kspdg.pe1.e1_envs import PE1_E1_I1_Env
from kspdg.pe1.e1_envs import PE1_E1_I4_Env

from os.path import join, dirname
from dotenv import load_dotenv

from sliding_window import SlidingWindow

# Load configuration from .env
#dotenv_path = join(dirname(__file__), 'arclib_mit', 'agents', '.env')
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

# Set OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]

# Load configuration from alex_prompts.txt
#dotenv_path = join(dirname(__file__), 'arclib_mit', 'agents', 'alex_prompts.txt')
dotenv_path = join(dirname(__file__), 'alex_prompts.txt')
load_dotenv(dotenv_path)

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

        # Sliding window
        self.sliding_window = SlidingWindow(int(os.environ['SLIDING_WINDOW_SIZE']))

        # Create a stream to log actions
        log_name = "./logs/fine_tune_agent_log_" + self.scenario + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'
        self.log = open(log_name, mode='w', newline='')
        head = ['throttles', 'duration', 'time', 'vehicle_mass', 'vehicle_propellant', 'pursuer_pos_x', 'pursuer_pos_y', 'pursuer_pos_z', 'pursuer_vel_x', 'pursuer_vel_y', 'pursuer_vel_z', 'evader_pos_x', 'evader_pos_y', 'evader_pos_z', 'evader_vel_x', 'evader_vel_y', 'evader_vel_z']
        csv.writer(self.log).writerow(head)


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

    def save_action(self, observation, action):
        """
            Saves the actions to a dictionary, importing the data from the observation and the activated throttles.
            Args:
                observation: observation from the environment
                action
        """
        row = observation
        row.insert(0, action[3])
        row.insert(0, action[0:3])
        csv.writer(self.log).writerow(row)
        self.log.flush()

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
        self.evaluation (observation)

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

        # Generate prompt
        messages = []
        # Add system prompt if this is the first completion
        if self.first_completion:
            messages.append({"role": "system", "content": os.environ['SYSTEM_PROMPT']})
        # Append messages from the sliding window
        for item in self.sliding_window.get_messages():
            messages.append(item)
        # Add user prompt for current observation
        messages.append({"role": "user", "content": message})
        prompt = messages

        try:
            action = self.check_response(response=self.get_completion(prompt=prompt, model=self.model))
            self.save_action(observation, action)
        except Exception as e:
            print ("Exception: " + str(e))
            action = [0,0,0,0.1]

        # Add response to sliding window
        self.sliding_window.append(prompt=message,
                                   arguments="{\n  \"ft\": " + str(action[0]) + ",\n  \"rt\": " + str(action[1]) + ",\n  \"dt\": " + str(action[2]) + "\n}")

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
        #   "{-1, 0, 0}
        # Transform to the first form
        index = function_args.find("{")
        if index == -1:
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
        print("Response message: " + str(response))
        if response.get("function_call"):
            duration = 0.5
            available_functions = {
                "perform_action": lambda ft, rt, dt: [ft, rt, dt, duration],
            }
            function_name = response["function_call"]["name"]
            if function_name not in available_functions:
                print("error: LLM called wrong function, name:", function_name)
                function_response = [0,0,0,0.1]
            else:
                function_to_call = available_functions[function_name]

                # Get function arguments
                old_function_args = response["function_call"]["arguments"]
                function_args = self.clean_response(response["function_call"]["arguments"])
                try:
                    function_args = json.loads(function_args)
                    function_response = function_to_call(**function_args)
                except Exception as ex:
                    print (function_args)
                    self.clean_response(old_function_args)
                    function_response = [0,0,0,0.1]
        else:
            print("error: LLM did not call function")
            function_response = [0, 0, 0, 0.1]

        return function_response

    def get_completion(self, prompt, model="ft:gpt-3.5-turbo-1106:personal::8MFqjElw"):

        print ("Prompt:")
        for message in prompt:
            print (message)

        # Perform chat completion
        time_before = time.time()
        response = openai.ChatCompletion.create(
            model=model,
            messages=prompt,
            functions=self.functions,
#            max_tokens = 100, # limit output tokens (enough for valid responses)
            temperature=0  # randomness, cool approach if we want to adjust some param with this
        )
        time_after = time.time()
        print ("Chat completion took " + str(time_after - time_before) + " seconds")
        self.first_completion = False

        return response.choices[0].message


if __name__ == "__main__":

    scenario = os.environ['SCENARIO']

    scenarios = dict()
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
