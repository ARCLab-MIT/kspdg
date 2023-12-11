import ast
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv

#dotenv_path = join(dirname(__file__), '..\\agents\.env')
dotenv_path = os.path.join(os.path.dirname(__file__), '..', 'agents', '.env')
load_dotenv(dotenv_path)

# Load configuration from alex_prompts.txt
#dotenv_path = join(dirname(__file__), 'arclib_mit', 'agents', 'alex_prompts.txt')
dotenv_path = os.path.join(os.path.dirname(__file__), '..', 'agents', 'alex_prompts.txt')
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

def evaluate_angle_distance(angle, distance):
    angle = np.abs(angle)
    if angle < 90:
        angle_gauge = "extremely poor"
    elif angle < 160:
        angle_gauge = "poor"
    elif angle < 170:
        angle_gauge = "average"
    elif angle < 175:
        angle_gauge = "good"
    else:
        angle_gauge = "excellent"

    if distance < 50:
        distance_gauge = "good"
    elif distance < 150:
        distance_gauge = "excellent"
    elif distance < 200:
        distance_gauge = "good"
    elif distance < 500:
        distance_gauge = "average"
    elif distance < 1000:
        distance_gauge = "poor"
    else:
        distance_gauge = "extremely poor"

    return angle_gauge, distance_gauge

def translate_action(action):
    result = ["none", "none", "none"]

    if action[0] == -1:
        result[0] = 'backward'
    elif action[0] == 1:
        result[0] = 'forward'

    if action[1] == -1:
        result[1] = 'left'
    elif action[1] == 1:
        result[1] = 'right'

    if action[2] == -1:
        result[2] = 'down'
    elif action[2] == 1:
        result[2] = 'up'

    result = ["\"" + item + "\"" for item in result]

    return result

def csv_to_json_for_history(csv_file_path : str, problem_env : str, use_short_names: bool = False, use_relative_coordinates: bool = False, use_enum: bool = True, size: int = 4, stride: int = 1):

    if problem_env.startswith('sb'):
        if use_short_names:
            sys.exit("Short names not supported for SB scenario")

    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Transform each row into the required JSON structure
    json_list = [] # List of JSON structures

    # JSON history
    json_history = []

    # Assistant content
    assistant_content = None
    if "ASSISTANT_CONTENT" in os.environ:
        assistant_content = os.environ["ASSISTANT_CONTENT"]

    """
    # Append system prompt at the beginning
    if problem_env.startswith('pe'):
        system_message_structure = {
            "messages": [
                {"role": "system", "content": os.environ['PE_SYSTEM_PROMPT']},
                {"role": "assistant", "content": "Understood."}
            ]
        }
        json_list.append(system_message_structure)
    elif problem_env.startswith('sb'):
        system_message_structure = {
            "messages": [
                {"role": "system", "content": os.environ['SB_SYSTEM_PROMPT']},
                {"role": "assistant", "content": "Understood."}
            ]
        }
        json_list.append(system_message_structure)
    """

    pos = 0
    skip_null_actions = False
    for _, row in df.iterrows():
        input_data = {k: v for k, v in row.items() if k != 'throttles' and k != 'next_throttles'}
        # Eliminate time
        del input_data['time']

        # Calculate distance and alignment angle
        rel_position = [input_data['evader_pos_x'] - input_data['pursuer_pos_x'],
                        input_data['evader_pos_y'] - input_data['pursuer_pos_y'],
                        input_data['evader_pos_z'] - input_data['pursuer_pos_z']]
        distance = np.linalg.norm(rel_position, ord=2)

        if filename.startswith('sb'):
            # Calculate alignment angle
            p_e_position = [input_data['evader_pos_x'] - input_data['pursuer_pos_x'],
                            input_data['evader_pos_y'] - input_data['pursuer_pos_y'],
                            input_data['evader_pos_z'] - input_data['pursuer_pos_z']]
            p_s_position = [input_data['sun_pos_x'] - input_data['pursuer_pos_x'],
                            input_data['sun_pos_y'] - input_data['pursuer_pos_y'],
                            input_data['sun_pos_z'] - input_data['pursuer_pos_z']]
            alignment_angle = math.degrees(np.arccos(np.dot(p_e_position, p_s_position) / (np.linalg.norm(p_e_position, ord=2) * np.linalg.norm(p_s_position, ord=2))))

        if use_relative_coordinates:
            # Add relative position and velocity
            input_data['relative_pos_x'] = input_data['evader_pos_x'] - input_data['pursuer_pos_x']
            input_data['relative_pos_y'] = input_data['evader_pos_y'] - input_data['pursuer_pos_y']
            input_data['relative_pos_z'] = input_data['evader_pos_z'] - input_data['pursuer_pos_z']

            input_data['relative_vel_x'] = input_data['evader_vel_x'] - input_data['pursuer_vel_x']
            input_data['relative_vel_y'] = input_data['evader_vel_y'] - input_data['pursuer_vel_y']
            input_data['relative_vel_z'] = input_data['evader_vel_z'] - input_data['pursuer_vel_z']

            if filename.startswith('sb'):
                input_data['relative_sun_pos_x'] = input_data['sun_pos_x'] - input_data['pursuer_pos_x']
                input_data['relative_sun_pos_y'] = input_data['sun_pos_y'] - input_data['pursuer_pos_y']
                input_data['relative_sun_pos_z'] = input_data['sun_pos_z'] - input_data['pursuer_pos_z']

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

            if filename.startswith('sb'):
                del input_data['sun_pos_x']
                del input_data['sun_pos_y']
                del input_data['sun_pos_z']

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

        # Replace names for SB scenario
        if filename.startswith('sb'):
            if use_relative_coordinates:
                input_data["relative_pos"] = [input_data.pop('relative_pos_x'), input_data.pop('relative_pos_y'), input_data.pop('relative_pos_z')]
                input_data["relative_vel"] = [input_data.pop('relative_vel_x'), input_data.pop('relative_vel_y'), input_data.pop('relative_vel_z')]
                input_data["relative_sun_pos"] = [input_data.pop('relative_sun_pos_x'), input_data.pop('relative_sun_pos_y'), input_data.pop('relative_sun_pos_z')]

            else:
                pursuer_pos_x = input_data.pop('pursuer_pos_x')
                pursuer_pos_y = input_data.pop('pursuer_pos_y')
                pursuer_pos_z = input_data.pop('pursuer_pos_z')
                input_data["vessel_pos"] = [pursuer_pos_x, pursuer_pos_y, pursuer_pos_z]

                pursuer_vel_x = input_data.pop('pursuer_vel_x')
                pursuer_vel_y = input_data.pop('pursuer_vel_y')
                pursuer_vel_z = input_data.pop('pursuer_vel_z')
                input_data["vessel_vel"] = [pursuer_vel_x, pursuer_vel_y, pursuer_vel_z]

                evader_pos_x = input_data.pop('evader_pos_x')
                evader_pos_y = input_data.pop('evader_pos_y')
                evader_pos_z = input_data.pop('evader_pos_z')
                input_data["satellite_pos"] = [evader_pos_x, evader_pos_y, evader_pos_z]

                evader_vel_x = input_data.pop('evader_vel_x')
                evader_vel_y = input_data.pop('evader_vel_y')
                evader_vel_z = input_data.pop('evader_vel_z')
                input_data["satellite_vel"] = [evader_vel_x, evader_vel_y, evader_vel_z]

                sun_pos_x = input_data.pop('sun_pos_x')
                sun_pos_y = input_data.pop('sun_pos_y')
                sun_pos_z = input_data.pop('sun_pos_z')
                input_data["sun_pos"] = [sun_pos_x, sun_pos_y, sun_pos_z]

        output_label = row['throttles']
        output_label = ast.literal_eval(output_label)
        # Translate action
        if use_enum:
            output_label = translate_action(output_label)

        if problem_env.startswith('pe'):
            message_structure = {
                "messages": [
                    {"role": "user", "content": os.environ['PE_USER_PROMPT'] + json.dumps(input_data)},
                    {"role": "assistant", "content": assistant_content, "function_call": {"name": "perform_action",
                                                                              "arguments": "{\"ft\": " + str(output_label[0]) + ", \"rt\": " +
                                                                                                         str(output_label[1]) + ", \"dt\": " +
                                                                                                         str(output_label[2]) + "}"}}
                ]
            }
        elif problem_env.startswith('sb'):
            # Evaluate distance and alignment angle
            angle_gauge, distance_gauge = evaluate_angle_distance(alignment_angle, distance)
            chain_of_thought = os.environ['SB_CHAIN_OF_THOUGHT'].format(angle_gauge, alignment_angle, distance_gauge, distance)
            question = os.environ['SB_USER_PROMPT'].format(json.dumps(input_data))
            message_structure = {
                "messages": [
#                    {"role": "system", "content": os.environ['SB_SYSTEM_PROMPT']},
                    {"role": "user", "content": chain_of_thought + question},
                    {"role": "assistant", "content": assistant_content, "function_call": {"name": "perform_action",
                                                                             "arguments": "{\"ft\": " + str(output_label[0]) + ", \"rt\": " +
                                                                                                        str(output_label[1]) + ", \"dt\": " +
                                                                                                        str(output_label[2]) + "}"}}
                ]
            }

        # Add message to history with initial padding to ensure sliding window is full
        if len(json_history) == 0:
            output_label = [0, 0, 0]
            if use_enum:
                output_label = translate_action(output_label)

            padding  = message_structure.copy()
            padding['messages'][1] = {"role": "assistant",
                                          "content": assistant_content,
                                          "function_call": {"name": "perform_action",
                                                            "arguments": "{\"ft\": " + str(output_label[0]) + ", \"rt\": " +
                                                                        str(output_label[1]) + ", \"dt\": " +
                                                                        str(output_label[2]) + "}"}}
            for i in range(sliding_window_size):
                json_history.append(padding)

        json_history.append(message_structure)

        # Skip null actions except the first one in a sequence of actions
        if row['throttles'] == '[0, 0, 0]':
            continue
        """
            if skip_null_actions:
                continue
            skip_null_actions = True
        else:
            skip_null_actions = False
        """

        # Add message to json list
        messages_in_window = [{"role": "system", "content": os.environ['SB_SYSTEM_PROMPT']}]
        start = -(sliding_window_size+1)
        if len(json_history) < -start:
            start = 0
        for item in json_history[start:]:
           messages_in_window = messages_in_window + item['messages']
        message_structure = {
            "messages": messages_in_window
        }
        json_list.append(message_structure)

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
