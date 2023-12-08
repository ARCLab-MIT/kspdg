"""install pynput"""
from pynput import keyboard
from kspdg.agent_api.base_agent import KSPDGBaseAgent
from kspdg.pe1.e1_envs import PE1_E1_I4_Env
from kspdg.pe1.e1_envs import PE1_E1_I3_Env
from kspdg.pe1.e1_envs import PE1_E1_I2_Env
from kspdg.pe1.e1_envs import PE1_E1_I1_Env
from kspdg.pe1.e3_envs import PE1_E3_I3_Env, PE1_E3_I2_Env
from kspdg.pe1.e4_envs import PE1_E4_I2_Env
from kspdg.sb1.e1_envs import SB1_E1_I5_Env
from kspdg.sb1.e1_envs import SB1_E1_I1_Env

from kspdg.agent_api.runner import AgentEnvRunner

import krpc

import numpy as np
import sys
import math

# New imports from the original code
import csv
import datetime

debug = False

def write_dict_to_csv(d, filename):
    """
    Write a dictionary to a csv file
    Args:
        d: dictionary to write
        filename: name of the file to write to
    """
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=d.keys())
        writer.writeheader()
        for i in range(len(next(iter(d.values())))):
            row = {k: d[k][i] for k in d.keys()}
            writer.writerow(row)

class State():
    def __init__(self, observation, sun_position):
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

        self.sun_position = sun_position
        p_e_pos = self.evader_position - self.pursuer_position
        p_s_pos = self.sun_position - self.pursuer_position
        angle_cosine = np.dot(p_e_pos, p_s_pos) / (np.linalg.norm(p_e_pos) * np.linalg.norm(p_s_pos))
        self.alignment_angle = math.degrees(np.arccos(angle_cosine))

class KeyboardControlledAgent(KSPDGBaseAgent):
    """
    Controls:
    H/N: Forward/Backward
    J/L: Left/Right
    I/K: Down/Up
    (controls are taken from here: https://wiki.kerbalspaceprogram.com/wiki/Key_bindings)

    There might be a lag between controls and the game

    Disable the controls in game, or press the controls on another focused window
    """
    def __init__(self, scenario):
        super().__init__()

        self.scenario = scenario
        self.forward_throttle = 0
        self.right_throttle = 0
        self.down_throttle = 0
        self.actions_dict = {                   # Names for all the different columns to append data to, useful for the csv export.
            'throttles': [],
            'time': [],
            'vehicle_mass': [],
            'vehicle_propellant': [],
            'pursuer_pos_x': [],
            'pursuer_pos_y': [],
            'pursuer_pos_z': [],
            'pursuer_vel_x': [],
            'pursuer_vel_y': [],
            'pursuer_vel_z': [],
            'evader_pos_x': [],
            'evader_pos_y': [],
            'evader_pos_z': [],
            'evader_vel_x': [],
            'evader_vel_y': [],
            'evader_vel_z': [],
            'sun_pos_x': [],
            'sun_pos_y': [],
            'sun_pos_z': [],
        }

        """ Connect to the KRPC server.
        """
        self.conn = krpc.connect()

        """ Get the active vessel and celestial body
        """
        self.vessel = self.conn.space_center.active_vessel
        self.body = self.vessel.orbit.body

        listener = keyboard.Listener(on_press=self.on_key_press, on_release=self.on_key_release)
        listener.start()

    def on_key_press(self, key):
        try:
            if debug:
                print("key pressed:", key.char)
            if key.char == 'h':
                self.forward_throttle = 1.0
            elif key.char == 'n':
                self.forward_throttle = -1.0
            elif key.char == 'j':
                self.right_throttle = -1.0
            elif key.char == 'l':
                self.right_throttle = 1.0
            elif key.char == 'i':
                self.down_throttle = 1.0
            elif key.char == 'k':
                self.down_throttle = -1.0
        except AttributeError:
            pass

    def on_key_release(self, key):
        try:
            if debug:
                print("key released:", key.char)
            if key.char in ('h', 'n'):
                self.forward_throttle = 0
            elif key.char in ('j', 'l'):
                self.right_throttle = 0
            elif key.char in ('i', 'k'):
                self.down_throttle = 0
        except AttributeError:
            pass

    def save_actions(self, observation):
        """
            Saves the actions to a dictionary, importing the data from the observation and the activated throttles.
            Args:
                observation: observation from the environment
        """
        keys = list(self.actions_dict.keys())
        keys.remove('throttles')
        self.actions_dict['throttles'].append([self.forward_throttle, self.right_throttle, self.down_throttle])
        for i, key in enumerate(keys):
            if key in ['sun_pos_x', 'sun_pos_y', 'sun_pos_z']:
                continue
            self.actions_dict[key].append(observation[i])

        """ add sun position for SB (sun-blocking) scenarios
        """
        if self.scenario.startswith('SB'):
            self.actions_dict['sun_pos_x'].append(observation[-3])
            self.actions_dict['sun_pos_y'].append(observation[-2])
            self.actions_dict['sun_pos_z'].append(observation[-1])

            # Show alignment angle
            sun_position = [observation[-3], observation[-2], observation[-1]]
            state = State(observation, sun_position)
            print("Evader-Pursuer-Sun angle (degrees): ", state.alignment_angle)

        if debug:
            print(self.actions_dict)
            
    def get_action(self, observation):
        """ compute agent's action given observation
        This function is necessary to define as it overrides 
        an abstract method
        """

        """ add sun position for SB (sun-blocking) scenarios
        """
        if self.scenario.startswith('SB'):
            reference_frame = self.body.orbital_reference_frame
            sun_pos = self.conn.space_center.bodies['Sun'].position(reference_frame)
            observation.append(sun_pos[0])
            observation.append(sun_pos[1])
            observation.append(sun_pos[2])

        # Return action list
        print(self.forward_throttle, self.right_throttle, self.down_throttle)
        self.save_actions(observation)
        return [self.forward_throttle, self.right_throttle, self.down_throttle, 0.5]

if __name__ == "__main__":
    try:
        scenario = 'SB1_E1_I5'
        env = SB1_E1_I5_Env

        keyboard_agent = KeyboardControlledAgent(scenario)
        runner = AgentEnvRunner(
            agent=keyboard_agent,
            env_cls=env,
            env_kwargs=None,
            runner_timeout=240,
            # debug=True
            debug=False
            )
        runner.run()
    except Exception as e:
        print("Something went wrong: " + str(e))
    finally:
        print("Saving data to csv...")
        if(len(keyboard_agent.actions_dict['throttles']) > 10):
            write_dict_to_csv(keyboard_agent.actions_dict, '../agents_data/pe1_e3_i2_keyboard_agent_actions_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv')
            print("Success!" + '../agents_data/pe1_e3_i2_keyboard_agent_actions_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv")
