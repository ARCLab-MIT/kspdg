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
from kspdg.sb1.e1_envs import SB1_E1_I1_Env
from kspdg.sb1.e1_envs import SB1_E1_I2_Env
from kspdg.sb1.e1_envs import SB1_E1_I3_Env
from kspdg.sb1.e1_envs import SB1_E1_I4_Env
from kspdg.sb1.e1_envs import SB1_E1_I5_Env

from os.path import join, dirname
from dotenv import load_dotenv

from sliding_window import SlidingWindow

"""
from arclab_mit.agents.extended_obs_agent.simulate import closest_approach
from arclab_mit.agents.common import obs_to_state, state_to_message
"""

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

class State():
    def __init__(self, observation):
        self.mission_time = observation[0]
        self.vehicle_mass = observation[1]
        self.vehicle_propellant = observation[2]

        self.pursuer_position = np.array([observation[3], observation[4], observation[5]])
        self.pursuer_velocity = np.array([observation[6], observation[7], observation[8]])

        self.evader_position = np.array([observation[9], observation[10], observation[11]])
        self.evader_velocity = np.array([observation[12], observation[13], observation[14]])

        self.rel_position = self.evader_position - self.pursuer_position
        self.rel_velocity = self.evader_velocity - self.pursuer_velocity

        self.distance = np.linalg.norm(self.rel_position, ord=2)
        self.velocity = np.linalg.norm(self.rel_velocity, ord=2)

    def distance_to_target(self):
        return self.distance

    def velocity_to_target(self):
        return self.velocity


def rotate_vector(vector, axis, angle):
    # Rotate a vector around an axis by a specified angle (in radians)
    axis = axis / np.linalg.norm(axis)
    q = np.append(np.cos(angle / 2), axis * np.sin(angle / 2))
    rotation_matrix = np.array([
        [1 - 2 * (q[2]**2 + q[3]**2), 2 * (q[1]*q[2] - q[0]*q[3]), 2 * (q[1]*q[3] + q[0]*q[2])],
        [2 * (q[1]*q[2] + q[0]*q[3]), 1 - 2 * (q[1]**2 + q[3]**2), 2 * (q[2]*q[3] - q[0]*q[1])],
        [2 * (q[1]*q[3] - q[0]*q[2]), 2 * (q[2]*q[3] + q[0]*q[1]), 1 - 2 * (q[1]**2 + q[2]**2)]
    ])
    rotated_vector = np.dot(rotation_matrix, vector)
    return rotated_vector

def calculate_prograde_vector (initial, target):

    # Calculate the pitch angle (angle between initial and target to)
    pitch_angle = np.arccos(np.dot(initial, target) / (np.linalg.norm(initial) * np.linalg.norm(target)))

    # Perform pitch adjustment
    axis_of_rotation = np.cross(initial, target)
    prograde_vector = rotate_vector(initial, axis_of_rotation, pitch_angle)

    # print("Aligned Prograde Vector:", prograde_vector)
    return prograde_vector

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix
class StateHistory():
    def __init__(self):
        self.history = []

    def append(self, observation):
        self.history.append(observation)

    def len(self):
        return len(self.history)
    def get_at(self, pos):
        return self.history[pos] if pos < len(self.history) else None

    def get_lastState(self):
        return self.history[-1] if len(self.history) > 0 else None

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

        self.stateHistory = StateHistory()

        self.first_completion = True
        self.closest_distance = sys.float_info.max
        self.use_relative_coordinates = (os.environ['USE_RELATIVE_COORDINATES'].lower() == "true")
        self.use_short_argument_names = (os.environ['USE_SHORT_ARGUMENT_NAMES'].lower() == "true")

        # Connect to the KRPC server.
        self.conn = vessel = krpc.connect()
        # Get the active vessel
        self.vessel = self.conn.space_center.active_vessel
        # Get the celestial body
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

        # Interval between actions
        self.duration = 0.5

        # Threshold for prograde alignment
        self.THRESHOLD = 0.1

    # Return sun position in the celestial body orbital reference frame
    def get_sunPosition(self):
        reference_frame = self.body.orbital_reference_frame
        # Get the sun position in the given reference frame
        sun_pos = self.conn.space_center.bodies['Sun'].position(reference_frame)
        return sun_pos

    def evaluation (self, observation):
        state = State(observation)
        distance = state.distance_to_target()
        if (distance < self.closest_distance):
            self.closest_distance = distance
        print(f'Closest Distance: {self.closest_distance:.2f}')
        print(f'Distance: {state.distance:.2f}')
        print(f'Velocity: {state.velocity:.2f}')
        print(f'Mission time: {state.mission_time:.2f}')
        print(f'Fuel: {state.vehicle_propellant:.2f}')
        print("Sun location: " + str(self.get_sunPosition()))
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
        # and confirm that these values are close to pursuer position and velocity from observation considering that y-axis has opposite sign
        # due to the fact the kRPC uses left-handed coordinate system. Differences are due to the movement of the vessel reference frame.

        """ compute agent's action given observation """
        print("get_action called, prompting ChatGPT...")

        state = State(observation)
        self.stateHistory.append(state)

        self.evaluation (observation)

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
                    str(f"\"rx\": {state.rel_position[0]}, ") + \
                    str(f"\"ry\": {state.rel_position[1]}, ") + \
                    str(f"\"rz\": {state.rel_position[2]}, ") + \
                    str(f"\"rvx\": {state.rel_velocity[0]}, ") + \
                    str(f"\"rvy\": {state.rel_velocity[1]}, ") + \
                    str(f"\"rvz\": {state.rel_velocity[2]}") + \
                    "}"
            else:
                message += \
                    str(f"\"vehicle_mass\": {observation[1]}, ") + \
                    str(f"\"vehicle_propellant\": {observation[2]}, ") + \
                    str(f"\"relative_pos_x\": {state.rel_position[0]}, ") + \
                    str(f"\"relative_pos_y\": {state.rel_position[1]}, ") + \
                    str(f"\"relative_pos_z\": {state.rel_position[2]}, ") + \
                    str(f"\"relative_vel_x\": {state.rel_velocity[0]}, ") + \
                    str(f"\"relative_vel_y\": {state.rel_velocity[1]}, ") + \
                    str(f"\"relative_vel_z\": {state.rel_velocity[2]}") + \
                    "}"
        else:
            if self.use_short_argument_names:
                message += \
                    str(f"\"m\": {observation[1]}, ") + \
                    str(f"\"f\": {observation[2]}, ") + \
                    str(f"\"px\": {pursuer_position[0]}, ") + \
                    str(f"\"py\": {pursuer_position[1]}, ") + \
                    str(f"\"pz\": {pursuer_position[2]}, ") + \
                    str(f"\"pvx\": {pursuer_velocity[0]}, ") + \
                    str(f"\"pvy\": {pursuer_velocity[1]}, ") + \
                    str(f"\"pvz\": {pursuer_velocity[2]}, ") + \
                    str(f"\"ex\": {state.evader_position[0]}, ") + \
                    str(f"\"ey\": {state.evader_position[1]}, ") +  \
                    str(f"\"ez\": {state.evader_position[2]}, ") + \
                    str(f"\"evx\": {state.evader_velocity[0]}, ") + \
                    str(f"\"evy\": {state.evader_velocity[1]}, ") + \
                    str(f"\"evz\": {state.evader_velocity[2]}") + \
                    "}"
            else:
                message += \
                    str(f"\"vehicle_mass\": {observation[1]}, ") + \
                    str(f"\"vehicle_propellant\": {observation[2]}, ") + \
                    str(f"\"pursuer_pos_x\":  {state.pursuer_position[0]}, ") + \
                    str(f"\"pursuer_pos_y\":  {state.pursuer_position[1]}, ") + \
                    str(f"\"pursuer_pos_z\":  {state.pursuer_position[2]}, ") + \
                    str(f"\"pursuer_vel_x\":  {state.pursuer_velocity[0]}, ") + \
                    str(f"\"pursuer_vel_y\":  {state.pursuer_velocity[1]}, ") + \
                    str(f"\"pursuer_vel_z\":  {state.pursuer_velocity[2]}, ") + \
                    str(f"\"evader_pos_x\":  {state.evader_position[0]}, ") + \
                    str(f"\"evader_pos_y\":  {state.evader_position[1]}, ") +  \
                    str(f"\"evader_pos_z\":  {state.evader_position[2]}, ") + \
                    str(f"\"evader_vel_x\":  {state.evader_velocity[0]}, ") + \
                    str(f"\"evader_vel_y\":  {state.evader_velocity[1]}, ") + \
                    str(f"\"evader_vel_z\":  {state.evader_velocity[2]}") + \
                    "}"

        # Generate prompt
        messages = []
        messages.append({"role": "system", "content": os.environ['SYSTEM_PROMPT']})
        # Add system prompt if this is the first completion
        """
        if self.first_completion:
            messages.append({"role": "system", "content": os.environ['SYSTEM_PROMPT']})
        """
        # Append messages from the sliding window
        for item in self.sliding_window.get_messages():
            messages.append(item)

        """"
        # Add hint to query best action for current observation
        # Estimate closest approach and time when both vehicles stop accelerating
        closest_state, closest_time = closest_approach(obs_to_state(observation), 30)

        hint = str(f"If your spacecraft and the evader both stop accelerating, this will be the simulated closest approach (happens at time {closest_time}s): {state_to_message(closest_state)}\n")
        messages.append({"role": "user", "content": hint})
        """

        # Add user prompt to query best action for current observation
        messages.append({"role": "user", "content": message})
        prompt = messages

        try:
            action = self.check_response(response=self.get_completion(prompt=prompt, model=self.model))
            self.save_action(observation, action)
        except Exception as e:
            print ("Exception: " + str(e))
            action = [0,0,0,0.1]

        if action[3] == self.duration:
            # If throttle is [0, 0, 0] and
            if action[0] == 0 and action[1] == 0 and action[2] == 0:
                if self.scenario.startswith('PE1'):
                    # Add randomness to align prograde vector with relative position
                    random_value = random.random()
                    if random_value >= 0.5:
                        vel_pos_frame = calculate_prograde_vector(state.rel_velocity, state.rel_position)
                        vel_pos_frame = vel_pos_frame / np.linalg.norm(vel_pos_frame)

                        if state.velocity > 50:
                            action[0] = -1 # backward
                        else:
                            action[0] = 1 # forward

                        # Rotate to align prograde vector with relative position
                        if abs(vel_pos_frame[1]) > self.THRESHOLD:
                            if vel_pos_frame[1] > 0:
                                action[1] = -1 # left
                            else:
                                action[1] = 1 # right
                        if abs(vel_pos_frame[1]) > self.THRESHOLD:
                            if vel_pos_frame[2] > 0:
                                action[2] = 1 # up
                            else:
                                action[1] = -1 # down

                elif self.scenario.startswith('SB1'):

                    # Add randomness to align prograde vector with relative position
                    random_value = random.random()
                    if random_value >= 0.5:
                        vel_pos_frame = calculate_prograde_vector(state.rel_velocity, self.get_sunPosition()-state.pursuer_position)
                        vel_pos_frame = vel_pos_frame / np.linalg.norm(vel_pos_frame)

                        """
                        if state.velocity > 50:
                            action[0] = -1  # backward
                        else:
                            action[0] = 1  # forward
                        """

                        # Rotate to align prograde vector with relative position
                        self.THRESHOLD = 1e-6
                        if abs(vel_pos_frame[1]) > self.THRESHOLD:
                            if vel_pos_frame[1] > 0:
                                action[1] = -1  # left
                            else:
                                action[1] = 1  # right
                        if abs(vel_pos_frame[1]) > self.THRESHOLD:
                            if vel_pos_frame[2] > 0:
                                action[2] = 1  # up
                            else:
                                action[1] = -1  # down

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
        print("Response message: " + str(response))
        if response.get("function_call"):
            duration = self.duration
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

    def get_completion(self, prompt, model="ft:gpt-3.5-turbo-1106:personal::8P6jHmTx"):

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
    scenarios["SB1_E1_I1"] = SB1_E1_I1_Env
    scenarios["SB1_E1_I2"] = SB1_E1_I2_Env
    scenarios["SB1_E1_I3"] = SB1_E1_I3_Env
    scenarios["SB1_E1_I4"] = SB1_E1_I4_Env
    scenarios["SB1_E1_I5"] = SB1_E1_I5_Env

    if not scenario in scenarios:
        print("Invalid scenario: " + scenario + " not in " + str(scenarios.keys()))
        sys.exit(1)

    print("Running scenario: " + scenario)

    my_agent = LLMAgent()
    runner = AgentEnvRunner(
        agent=my_agent,
        env_cls=scenarios[scenario],
        env_kwargs=None,
        runner_timeout=240,
        debug=False)
    runner.run()
