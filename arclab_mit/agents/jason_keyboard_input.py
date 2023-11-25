"""install pynput"""
from pynput import keyboard
from kspdg.agent_api.base_agent import KSPDGBaseAgent
from kspdg.pe1.e1_envs import PE1_E1_I3_Env
from kspdg.pe1.e1_envs import PE1_E1_I2_Env
from kspdg.agent_api.runner import AgentEnvRunner

# New imports from the original code
import csv
import datetime

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
    def __init__(self):
        super().__init__()
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
            'evader_vel_z': []
        }

        listener = keyboard.Listener(on_press=self.on_key_press, on_release=self.on_key_release)
        listener.start()

    def on_key_press(self, key):
        try:
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
            self.actions_dict[key].append(observation[i])
        print(self.actions_dict)
            
    def get_action(self, observation):
        """ compute agent's action given observation
        This function is necessary to define as it overrides 
        an abstract method
        """

        # Return action list
        print(self.forward_throttle, self.right_throttle, self.down_throttle)
        self.save_actions(observation)
        return [self.forward_throttle, self.right_throttle, self.down_throttle, 0.5]

if __name__ == "__main__":
    try:
        keyboard_agent = KeyboardControlledAgent()
        runner = AgentEnvRunner(
            agent=keyboard_agent,
            env_cls=PE1_E1_I2_Env,
            env_kwargs=None,
            runner_timeout=300,
            # debug=True
            debug=False
            )
        runner.run()
    except:
        print("Something went wrong")
    finally:
        print("Saving data to csv...")
        write_dict_to_csv(keyboard_agent.actions_dict, '../agents_data/pe1_i2_keyboard_agent_actions_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv')
        print("Success!" + '../agents_data/pe1_i2_keyboard_agent_actions_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv")
