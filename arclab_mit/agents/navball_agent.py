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

from kspdg.agent_api.base_agent import KSPDGBaseAgent
from kspdg.agent_api.runner import AgentEnvRunner

from kspdg.lbg1.lg2_envs import LBG1_LG2_I1_Env
from kspdg.lbg1.lg2_envs import LBG1_LG2_I2_Env
from kspdg.lbg1.lbg1_base import LadyBanditGuardGroup1Env
from kspdg.pe1.e1_envs import PE1_E1_I1_Env
from kspdg.pe1.e1_envs import PE1_E1_I2_Env
from kspdg.pe1.e1_envs import PE1_E1_I3_Env
from kspdg.pe1.e1_envs import PE1_E1_I4_Env
from kspdg.pe1.e3_envs import PE1_E3_I1_Env
from kspdg.pe1.e3_envs import PE1_E3_I2_Env
from kspdg.pe1.e3_envs import PE1_E3_I3_Env
from kspdg.pe1.e3_envs import PE1_E3_I4_Env
from kspdg.pe1.e4_envs import PE1_E4_I3_Env
from kspdg.pe1.pe1_base import PursuitEvadeGroup1Env
from kspdg.sb1.e1_envs import SB1_E1_I1_Env
from kspdg.sb1.e1_envs import SB1_E1_I2_Env
from kspdg.sb1.e1_envs import SB1_E1_I3_Env
from kspdg.sb1.e1_envs import SB1_E1_I4_Env
from kspdg.sb1.e1_envs import SB1_E1_I5_Env
from kspdg.sb1.sb1_base import SunBlockingGroup1Env


from os.path import join, dirname
from dotenv import load_dotenv

from arclab_mit.agents.sliding_window import SlidingWindow
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

APPROACH_SPEED = 40
VESSEL_ACCELERATION = 0.1
EVASION_DISTANCE = 0
ROTATION_THRESHOLD = 0.03

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
        self.use_prograde = (os.environ['USE_PROGRADE'].lower() == "true")

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
            log_name = "./logs/navball_log_" + self.scenario + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'
            self.log = open(log_name, mode='w', newline='')
            if self.scenario.lower().startswith("lbg"):
                if self.use_prograde:
                    head = ['throttles', 'duration', 'time', 'vehicle_mass', 'vehicle_propellant', 'pursuer_pos_x', 'pursuer_pos_y', 'pursuer_pos_z', 'pursuer_vel_x', 'pursuer_vel_y', 'pursuer_vel_z', 'evader_pos_x', 'evader_pos_y', 'evader_pos_z', 'evader_vel_x', 'evader_vel_y', 'evader_vel_z', 'guard_pos_x', 'guard_pos_y', 'guard_pos_z', 'guard_vel_x', 'guard_vel_y', 'guard_vel_z', 'prograde', 'weighted_score']
                else:
                    head = ['throttles', 'duration', 'time', 'vehicle_mass', 'vehicle_propellant', 'pursuer_pos_x', 'pursuer_pos_y', 'pursuer_pos_z', 'pursuer_vel_x', 'pursuer_vel_y', 'pursuer_vel_z', 'evader_pos_x', 'evader_pos_y', 'evader_pos_z', 'evader_vel_x', 'evader_vel_y', 'evader_vel_z', 'guard_pos_x', 'guard_pos_y', 'guard_pos_z', 'guard_vel_x', 'guard_vel_y', 'guard_vel_z', 'weighted_score']
            else:
                if self.use_prograde:
                    head = ['throttles', 'duration', 'time', 'vehicle_mass', 'vehicle_propellant', 'pursuer_pos_x', 'pursuer_pos_y', 'pursuer_pos_z', 'pursuer_vel_x', 'pursuer_vel_y', 'pursuer_vel_z', 'evader_pos_x', 'evader_pos_y', 'evader_pos_z', 'evader_vel_x', 'evader_vel_y', 'evader_vel_z', 'prograde', 'weighted_score']
                else:
                    head = ['throttles', 'duration', 'time', 'vehicle_mass', 'vehicle_propellant', 'pursuer_pos_x', 'pursuer_pos_y', 'pursuer_pos_z', 'pursuer_vel_x', 'pursuer_vel_y', 'pursuer_vel_z', 'evader_pos_x', 'evader_pos_y', 'evader_pos_z', 'evader_vel_x', 'evader_vel_y', 'evader_vel_z', 'weighted_score']

            csv.writer(self.log).writerow(head)

            log_name = log_name.replace("csv", "jsonl")
            self.log_jsonl = open(log_name, mode='w', newline='\n')

        # Interval between actions
        self.duration = 0.5

        # Sliding window parameters
        self.sliding_window_size = int(os.environ["SLIDING_WINDOW_SIZE"])
        self.sliding_window_stride = int(os.environ["SLIDING_WINDOW_STRIDE"])

        # Sliding window
        if self.scenario.startswith('PE'):
            self.sliding_window = SlidingWindow(self.sliding_window_size, 'PE',
                                                self.use_relative_coordinates,
                                                self.use_short_names, self.use_enum,
                                                self.use_prograde,
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

        self.debug = False

    # Return sun position in the celestial body orbital reference frame
    def get_sun_position(self):
        reference_frame = self.body.orbital_reference_frame
        # Get the sun position in the given reference frame
        sun_pos = self.conn.space_center.bodies['Sun'].position(reference_frame)
        return sun_pos

    def get_action(self, observation, sun_position=None):
        debug = self.debug

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
        #
        # This direction should be the same as the direction of the evader's position relative to pursuer's since vessel is pointing at evader.
        # y-axis (forward) in vessel's reference frame is pointing at target and x-axis to the right. Thus
        # direction [0, 1, 0] is transformed to rel_position direction in celestial body reference frame:
        #
        # self.conn.space_center.transform_direction([0, 1, 0], vessel_frame, celestial_body_frame)
        #

        """ compute agent's action given observation """
        print("get_action called, prompting ChatGPT model ..." + self.model)

        # Get the sun position in the given reference frame
        if sun_position is None:
            sun_position = self.get_sun_position()

        # Obtain reference frames
        surface_velocity_frame = self.vessel.surface_velocity_reference_frame
        surface_frame = self.vessel.surface_reference_frame

        vessel_frame = self.vessel.reference_frame
        celestial_body_frame = self.body.orbital_reference_frame

        # Build state and show it
        # Need to exchange vessel_up's y and z components since KSP uses left-handed coordinate system
        vessel_up = np.array(self.conn.space_center.transform_direction((0, 0, 1), vessel_frame, celestial_body_frame))
        vessel_up = State.lh_to_rh(vessel_up)
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

        print(f'Mission time: {state.mission_time:.2f}')
        print(f'Velocity: {state.velocity:.2f}')
        print(f'Closest distance: {self.closest_distance:.2f}')
        print(f'Weighted score: {self.weighted_score:.2f}')

        if debug:
            state.show()

        # Surface reference frame: x-axis points in the zenith (e.g. from the center of the body towards the center
        # of mass of the vessel; y-axis points north and tangential to the surface of the body (east); z-axis points
        # eastwards and tangential to the surface of the body (north).
        #
        # Celestial body orbital reference frame: x-axis points in an arbitrary direction through the equator (zenith),
        # y-axis points towards the north and z-axis points in an arbitrary direction through the equator.
        #
        # Rotation matrix from orbital to surface reference frame of the celestial body
        rot_matrix = state.surface_rot_matrix

        if debug:
            print("rot_matrix: " + str(rot_matrix))

            """ Check celestial to surface frame transformation
            """
            v = [1, 1, 1]
            p = np.array(self.conn.space_center.transform_direction(v, celestial_body_frame, surface_frame))
            q = np.matmul(v, rot_matrix)
            print("celestial to surface frame transformation: " + str(
                np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q))))

            """ Check vessel position is given in celestial body orbital reference frame
            """
            reference_frame = self.body.reference_frame
            p = np.array(self.conn.space_center.transform_position([0,0,0], vessel_frame, reference_frame))
            p = State.lh_to_rh(p)
            diff = (p - state.pursuer_position) / np.linalg.norm(state.pursuer_position)
            print("check pursuer position in body reference frame:" + str(diff) + " / " + str(np.linalg.norm(diff)))

            reference_frame = self.body.orbital_reference_frame
            p = np.array(self.conn.space_center.transform_position([0,0,0], vessel_frame, reference_frame))
            p = State.lh_to_rh(p)
            diff = (p - state.pursuer_position) / np.linalg.norm(state.pursuer_position)
            print("check pursuer position in orbital reference frame:" + str(diff) + " / " + str(np.linalg.norm(diff)))

            reference_frame = self.body.non_rotating_reference_frame
            p = np.array(self.conn.space_center.transform_position([0,0,0], vessel_frame, reference_frame))
            p = State.lh_to_rh(p)
            diff = (p - state.pursuer_position) / np.linalg.norm(state.pursuer_position)
            print("check pursuer position in non-rotating reference frame:" + str(diff) + " / " + str(np.linalg.norm(diff)))

            """ Check vessel velocity is given in celestial body orbital reference frame
            """
            reference_frame = self.body.reference_frame
            v = self.conn.space_center.transform_velocity([0, 0, 0], [0, 0, 0], vessel_frame, reference_frame)
            v = State.lh_to_rh(v)
            diff = (v - state.pursuer_velocity) / np.linalg.norm(state.pursuer_velocity)
            print("check pursuer velocity in body reference frame:" + str(diff) + " / " + str(np.linalg.norm(diff)))

            reference_frame = self.body.orbital_reference_frame
            v = self.conn.space_center.transform_velocity([0, 0, 0], [0, 0, 0], vessel_frame, reference_frame)
            v = State.lh_to_rh(v)
            diff = (v - state.pursuer_velocity) / np.linalg.norm(state.pursuer_velocity)
            print("check pursuer velocity in orbital reference frame:" + str(diff) + " / " + str(np.linalg.norm(diff)))

            reference_frame = self.body.non_rotating_reference_frame
            v = self.conn.space_center.transform_velocity([0, 0, 0], [0, 0, 0], vessel_frame, reference_frame)
            v = State.lh_to_rh(v)
            diff = (v - state.pursuer_velocity) / np.linalg.norm(state.pursuer_velocity)
            print("check pursuer velocity in non-rotating reference frame:" + str(diff) + " / " + str(np.linalg.norm(diff)))

            """ Check vessel is pointing at target
            """
            v = np.array(self.vessel.direction(celestial_body_frame))
            v = State.lh_to_rh(v)
            print("v: " + str(v))
            rel_position = state.rel_position
            print("rel_position: " + str(rel_position/np.linalg.norm(rel_position)))
            n = np.dot(v, rel_position) / (np.linalg.norm(v) * np.linalg.norm(state.rel_position))
            print("n: " + str(n))
            n = np.dot(v, state.vessel_up) / (np.linalg.norm(v) * np.linalg.norm(state.vessel_up))
            print("n_up: " + str(n))

            """ Obtain Euler angles of vessel's lever marker using these approaches:
                1) transforming directions from vessel to surface frame
                2) transforming directions from celestial body to surface frame using kprc
                3) transforming directions from celestial body to surface frame using the rotation matrix
            """
            vessel_up = np.array(self.conn.space_center.transform_direction((0, 0, 1), vessel_frame, surface_frame))
            v = np.array(self.vessel.direction(surface_frame))
            l_pitch, l_heading, l_roll = State.get_pitch_heading_roll(v, vessel_up)
            print(f"LEVER (using surface ref. frame) heading: {l_heading:.2f}, pitch: {l_pitch:.2f}, roll: {l_roll:.2f}")
            print("v: " + str(v))
            print('vessel_up: ' + str(vessel_up))

            vessel_up = State.rh_to_lh(state.vessel_up)
            vessel_up = np.array(self.conn.space_center.transform_direction(vessel_up, celestial_body_frame, surface_frame))
            l_pitch, l_heading, l_roll = State.get_pitch_heading_roll(v, vessel_up)
            print(f"LEVER (w/ stored vessel_up) heading: {l_heading:.2f}, pitch: {l_pitch:.2f}, roll: {l_roll:.2f}")
            print("v: " + str(v))
            print('vessel_up: ' + str(vessel_up))

            """ Obtain euler angles of lever marker (vessel nose direction)
            """
            v = State.rh_to_lh(state.rel_position)
            v = np.matmul(v, rot_matrix)
            vessel_up = State.rh_to_lh(state.vessel_up)
            vessel_up = np.matmul(vessel_up, rot_matrix)
            l_pitch, l_heading, l_roll = State.get_pitch_heading_roll(v, vessel_up)
            print(f"LEVER heading: {l_heading:.2f}, pitch: {l_pitch:.2f}, roll: {l_roll:.2f}")
            print("v: " + str(v))
            print('vessel_up: ' + str(vessel_up))

            """ Check vessel_up and rel_position are orthogonal
            """
            angle = State.angle_between_vectors(rel_position, state.vessel_up)
            print(f"angle between relative position and vessel up vectors {angle:.2f}")
            print(f"Determinant rotation matrix {np.linalg.det(state.vessel_rot_matrix):.5f}")

        if debug:
            """ Obtain angles of prograde's marker (direction of vessel velocity relative to target) using these approaches
                1) transforming directions from celestial body to surface frame using krpc
                2) transforming directions from celestial body to surface frame using rotation matrix
            """
            v = state.pursuer_velocity - state.evader_velocity
            v = State.rh_to_lh(v)
            v = self.conn.space_center.transform_direction(v, celestial_body_frame, surface_frame)
            vessel_up = np.array(self.conn.space_center.transform_direction((0, 0, 1), vessel_frame, surface_frame))
            p_pitch, p_heading, p_roll = State.get_pitch_heading_roll(v, vessel_up)
            print(f"PROGRADE (using surface ref. frame) heading: {p_heading:.2f}, pitch: {p_pitch:.2f}, roll: {p_roll:.2f}")
            print("v: " + str(v))
            print("vessel_up: " + str(state.vessel_up))

            """ Obtain euler angles of prograde marker (vessel's velocity relative to target)
            """
            p_pitch, p_heading, p_roll = state.get_prograde(ref_frame="surface", angles=True)
            print(f"PROGRADE heading: {p_heading:.2f}, pitch: {p_pitch:.2f}, roll: {p_roll:.2f}")

        if self.scenario.lower().startswith('pe'):
            """ Obtain evader velocity relative to pursuer in vessel reference frame (retrograde marker)
            """
            retrograde = state.get_retrograde(ref_frame="vessel")
            if debug:
                print(f"Retrograde direction is {retrograde}")

            print("retrograde: " + str(retrograde/np.linalg.norm(retrograde)))

            ft = 1
            v = retrograde / np.linalg.norm(retrograde)
            if (abs(v[0]) > ROTATION_THRESHOLD) or (abs(v[2]) > ROTATION_THRESHOLD):
                rt = 1 if retrograde[0] > 0 else -1
                dt = 1 if retrograde[2] > 0 else -1
            else:
                rt = 0
                dt = 0
#            if state.distance < 1500:
            if (np.dot(state.rel_position, state.rel_velocity) < 0) and (state.time_to_intercept * VESSEL_ACCELERATION < state.velocity) and (state.distance > EVASION_DISTANCE):
                # Target is approaching and intercept time is insufficient to stop
                if state.velocity > APPROACH_SPEED:
                    # Reduce speed
                    ft = -1
                else:
                    ft = 0
        elif self.scenario.lower.startswith('lbg'):
            """ Obtain evader velocity relative to pursuer in vessel reference frame (retrograde marker)
            """
            retrograde = state.get_retrorade()
            if debug:
                print(f"Retrograde direction is {retrograde}")

            """ Obtain guard marker in vessel reference frame
            """
            guard = State.rh_to_lh(state.guard_position)
            guard = np.matmul(guard, self.vessel_rot_matrix)
            if debug:
                print(f"Guard direction is {guard}")

            """ Forward acceleration """
            dist = 1500
            ft = 1
            if state.distance < dist:
                if state.velocity > 20:
                    ft = -1
                else:
                    ft = 0

            """ Rotate toward target
            """
            rt = 1 if retrograde[0] > 0 else -1
            dt = 1 if retrograde[2] > 0 else -1
            if guard[1] > 0:
                """ Evade guard """
                grt = 1 if guard[0] > 0 else -1
                gdt = 1 if guard[2] > 0 else -1
                if rt * grt > 0:
                    rt = -rt
                if dt * gdt > 0:
                    dt = -dt
        else:
            ft = 0
            rt = 0
            dt = 0

        # Add state to sliding window. Action is none since it is unknown at this moment
        self.sliding_window.add(state, None)

        ref_frame = 0
        if ref_frame == 0:
            # Burn vector in vessel reference frame
            burn_vec = [ft, rt, dt, self.duration]
            ref_frame = 0
        else:
            # Burn vector in celestial body orbital reference frame
            rot_matrix = np.linalg.inv(state.vessel_rot_matrix)
            v = [ft, rt, dt]
            v = np.matmul(v, rot_matrix)
            v = State.lh_to_rh(v)
            burn_vec = [v[0], v[1], v[2], self.duration]

            v = [0, 1, 0]
            v = np.matmul(v, rot_matrix)
            v = State.lh_to_rh(v)
            print("v: " + str(v))
            print("rel_position: " + str(state.rel_position/np.linalg.norm(state.rel_position, ord=2)))

        action = {
            "burn_vec": burn_vec,
            "ref_frame": ref_frame
        }

        """ Log result
        """
        if self.log is not None:
            row = observation
            row = list(row)
            row.insert(0, action["burn_vec"][3])
            row.insert(0, action["burn_vec"][0:3])
            csv.writer(self.log).writerow(row)
            self.log.flush()

        # Simulate latency of s seconds
        time.sleep(2)
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
                max_tokens = 150, # limit output tokens (enough for valid responses)
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
    scenarios["PE1_E3_I1"] = PE1_E3_I1_Env
    scenarios["PE1_E3_I2"] = PE1_E3_I2_Env
    scenarios["PE1_E3_I3"] = PE1_E3_I3_Env
    scenarios["PE1_E3_I4"] = PE1_E3_I4_Env
    scenarios["PE1_E4_I3"] = PE1_E4_I3_Env

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
