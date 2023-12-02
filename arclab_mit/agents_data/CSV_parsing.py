import pandas as pd
import json
import os
import sys
import random

import ast

from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '..\\agents\.env')
load_dotenv(dotenv_path)

pattern = r"pe\d+_i\d+_keyboard_agent_actions_\d{8}-\d{6}\.csv"
#csv_file_path = r".\arclab_mit\agents_data\""
csv_file_path = r".\""

def json_to_jsonl(input_file, output_file):
    with open(input_file, 'r') as infile:
        data = json.load(infile)

    with open(output_file, 'w') as outfile:
        for item in data:
            json.dump(item, outfile)
            outfile.write('\n')

def csv_to_json_for_history(csv_file_path : str, problem_env : str, use_short_names: bool = False, use_relative_coordinates: bool = False, size: int = 4, stride: int = 1):

    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Transform each row into the required JSON structure
    json_list = [] # List of JSON structures

    # JSON history
    json_history = []

    pos = 0
    for _, row in df.iterrows():
        """
        # Skip [0, 0, 0] actions
        if row['throttles'] == '[0, 0, 0]':
            continue
        """

        input_data = {k: v for k, v in row.items() if k != 'throttles' and k != 'next_throttles'}
        # Eliminate time
        del input_data['time']

        if use_relative_coordinates:
            # Add relative position and velocity
            input_data['relative_pos_x'] = input_data['evader_pos_x'] - input_data['pursuer_pos_x']
            input_data['relative_pos_y'] = input_data['evader_pos_y'] - input_data['pursuer_pos_y']
            input_data['relative_pos_z'] = input_data['evader_pos_z'] - input_data['pursuer_pos_z']

            input_data['relative_vel_x'] = input_data['evader_vel_x'] - input_data['pursuer_vel_x']
            input_data['relative_vel_y'] = input_data['evader_vel_y'] - input_data['pursuer_vel_y']
            input_data['relative_vel_z'] = input_data['evader_vel_z'] - input_data['pursuer_vel_z']

            # Delete unnecessary keys
            del input_data['pursuer_pos_x']
            del input_data['pursuer_pos_y']
            del input_data['pursuer_pos_z']
            del input_data['pursuer_vel_x']
            del input_data['pursuer_vel_y']
            del input_data['pursuer_vel_z']

            del input_data['evader_pos_x']
            del input_data['evader_pos_y']
            del input_data['evader_pos_z']
            del input_data['evader_vel_x']
            del input_data['evader_vel_y']
            del input_data['evader_vel_z']

        if use_short_names:
            input_data["m"] = input_data.pop("vehicle_mass")
            input_data["f"] = input_data.pop("vehicle_propellant")
            if use_relative_coordinates:
                input_data["rx"] = input_data.pop("relative_pos_x")
                input_data["ry"] = input_data.pop("relative_pos_y")
                input_data["rz"] = input_data.pop("relative_pos_z")
                input_data["rvx"] = input_data.pop("relative_vel_x")
                input_data["rvy"] = input_data.pop("relative_vel_y")
                input_data["rvz"] = input_data.pop("relative_vel_z")
            else:
                input_data["px"] = input_data.pop("pursuer_pos_x")
                input_data["py"] = input_data.pop("pursuer_pos_y")
                input_data["pz"] = input_data.pop("pursuer_pos_z")
                input_data["pvx"] = input_data.pop("pursuer_vel_x")
                input_data["pvy"] = input_data.pop("pursuer_vel_y")
                input_data["pvz"] = input_data.pop("pursuer_vel_z")
                input_data["ex"] = input_data.pop("evader_pos_x")
                input_data["ey"] = input_data.pop("evader_pos_y")
                input_data["ez"] = input_data.pop("evader_pos_z")
                input_data["evx"] = input_data.pop("evader_vel_x")
                input_data["evy"] = input_data.pop("evader_vel_y")
                input_data["evz"] = input_data.pop("evader_vel_z")

        output_label = row['throttles']
        output_label = ast.literal_eval(output_label)
        message_structure = {
            "messages": [
                {"role": "user", "content": "Best throttle to capture evader for " + json.dumps(input_data)},
                {"role": "assistant", "content": None, "function_call": {"name" : "perform_action", "arguments": "{\n  \"ft\": " + str(output_label[0]) + ",\n  \"rt\": " + str(output_label[1]) + ",\n  \"dt\": " + str(output_label[2]) + "\n}"}}
            ]
        }

        # Add message to history
        if len(json_history) == 0:
            # Fill history with copies of the first message but with throttles [0, 0, 0]
            initial_message_structure = {
                "messages": [
                    {"role": "user", "content": "Best throttle to capture evader for " + json.dumps(input_data)},
                    {"role": "assistant", "content": None, "function_call": {"name": "perform_action", "arguments": "{\n  \"ft\": 0,\n  \"rt\": 0,\n  \"dt\": 0\n}"}}
                ]
            }
            for i in range(sliding_window_size-1):
                json_history.append(initial_message_structure)
        json_history.append(message_structure)

        # Add message to json list
        # Hack to append only first item since OpenAI does not support lists in JSON lines just only dictionaries
        if len(json_history) < sliding_window_size:
            json_list.append(json_history[:][0])
        else:
            json_list.append(json_history[-sliding_window_size:][0])
        """
        if pos == 0:
            # Add message to json list
            if len(json_history) < sliding_window_size:
                json_list.append(json_history[:])
            else:
                json_list.append(json_history[-sliding_window_size:])
            pos = stride
        pos -= 1
        """

    # Save JSON to a file training and validation file
    json_file_path = csv_file_path.replace('.csv', '.json')
    with open(json_file_path, 'w') as file:
        json.dump(json_list, file, indent=4)
    print(f"JSON file saved: {json_file_path}")

if __name__== '__main__':

#    dotenv_path = join(dirname(_file_), '.env')
    # Load configuration from .env
    dotenv_path = "../agents/.env"
    load_dotenv(dotenv_path)

    use_relative_coordinates = (os.environ['USE_RELATIVE_COORDINATES'].lower() == "true")
    sliding_window_size = int(os.environ["SLIDING_WINDOW_SIZE"])
    sliding_window_stride = int(os.environ["SLIDING_WINDOW_STRIDE"])


    # Process all CSV files in the current directory
    problem_env = input("Enter the problem and environment (e.g., pe1_i3): ")
#    directory = '.'  # Current directory
    directory = '.'  # Current directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv") and filename.startswith(problem_env):
            csv_to_json_for_history(os.path.join(directory, filename), problem_env,
                                    use_relative_coordinates=use_relative_coordinates,
                                    size=sliding_window_size,
                                    stride=sliding_window_stride)
            file_prefix = os.path.join(directory, filename).replace('.csv', '')
            json_file = file_prefix + '.json'
            jsonl_file = file_prefix + '.jsonl'
            json_to_jsonl(json_file, jsonl_file)
