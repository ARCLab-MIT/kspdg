import pandas as pd
import json
import os
import numpy as np

import ast

from os.path import join, dirname
from dotenv import load_dotenv

from arclab_mit.agents.sliding_window import SlidingWindow
from arclab_mit.agents.agent_common import State, Action

dotenv_path = os.path.join(os.path.dirname(__file__), '..', 'agents', '.env')
load_dotenv(dotenv_path)

# Load configuration from alex_prompts.txt
dotenv_path = join(dirname(__file__), '..', 'agents', 'alex_prompts.txt')
load_dotenv(dotenv_path)

pattern = r"pe\d+_i\d+_keyboard_agent_actions_\d{8}-\d{6}\.csv"
# csv_file_path = r".\arclab_mit\agents_data\""
# csv_file_path = r".\""

DEFAULT_SKIP_ALL_NULL_ACTIONS = False
DEFAULT_SLIDING_WINDOW_STRIDE = False


def json_to_jsonl(input_file, output_file):
    with open(input_file, 'r') as infile:
        data = json.load(infile)

    with open(output_file, 'w') as outfile:
        for item in data:
            json.dump(item, outfile)
            outfile.write('\n')


def csv_to_json_for_history(csv_file_path: str, problem_env: str,
                            use_short_names: bool = State.DEFAULT_USE_SHORT_NAMES,
                            use_relative_coordinates: bool = State.DEFAULT_USE_RELATIVE_COORDINATES,
                            use_enum: bool = Action.DEFAULT_USE_ENUM,
                            use_prograde: bool = State.DEFAULT_USE_PROGRADE,
                            use_cot: bool = State.DEFAULT_USE_COT,
                            use_cot_speed_limit: bool = State.DEFAULT_USE_COT_SPEED_LIMIT,
                            size: int = SlidingWindow.DEFAULT_SLIDING_WINDOW_SIZE,
                            stride: int = DEFAULT_SLIDING_WINDOW_STRIDE,
                            embed_history: bool = SlidingWindow.DEFAULT_EMBED_HISTORY,
                            skip_all_null_actions: bool = DEFAULT_SKIP_ALL_NULL_ACTIONS):

    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    if problem_env.lower().startswith('pe'):
        sliding_window = SlidingWindow(size, 'PE',
                                       use_relative_coordinates, use_short_names, use_enum,
                                       use_prograde, use_cot, use_cot_speed_limit, embed_history,
                                       os.environ["PE_SYSTEM_PROMPT"],
                                       os.environ["PE_USER_PROMPT"],
                                       os.environ["PE_CHAIN_OF_THOUGHT"],
                                       os.environ["ASSISTANT_CONTENT"],
                                       os.environ["PE_HISTORY_PROMPT"],
                                       os.environ["PE_HISTORY_ITEM_PROMPT"])
    elif problem_env.lower().startswith('sb'):
        sliding_window = SlidingWindow(size, 'SB',
                                       use_relative_coordinates, use_short_names, use_enum,
                                       use_prograde, use_cot, use_cot_speed_limit, embed_history,
                                       os.environ["SB_SYSTEM_PROMPT"],
                                       os.environ["SB_USER_PROMPT"],
                                       os.environ["SB_CHAIN_OF_THOUGHT"],
                                       os.environ["ASSISTANT_CONTENT"],
                                       os.environ["SB_HISTORY_PROMPT"],
                                       os.environ["SB_HISTORY_ITEM_PROMPT"])
    elif problem_env.lower().startswith('lbg'):
        sliding_window = SlidingWindow(size, 'LBG',
                                       use_relative_coordinates, use_short_names, use_enum,
                                       use_prograde, use_cot, embed_history,
                                       os.environ["LBG_SYSTEM_PROMPT"],
                                       os.environ["LBG_USER_PROMPT"],
                                       os.environ["LBG_CHAIN_OF_THOUGHT"],
                                       os.environ["ASSISTANT_CONTENT"],
                                       os.environ["LBG_HISTORY_PROMPT"],
                                       os.environ["LBG_HISTORY_ITEM_PROMPT"])
    else:
        sliding_window = None

    # List of JSON structures
    json_list = []

    skip_null_actions = False
    for _, row in df.iterrows():
        input_data = {k: v for k, v in row.items() if k != 'throttles' and k != 'next_throttles'}
        
        if problem_env.lower().startswith('lbg'):
            observation = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            observation[0] = input_data["time"]
            observation[1] = input_data["bandit_mass"]
            observation[2] = input_data["bandit_propellant"]

            observation[3] = input_data['bandit_pos_x']
            observation[4] = input_data['bandit_pos_y']
            observation[5] = input_data['bandit_pos_z']

            observation[6] = input_data['bandit_vel_x']
            observation[7] = input_data['bandit_vel_y']
            observation[8] = input_data['bandit_vel_z']

            observation[9] = input_data['lady_pos_x']
            observation[10] = input_data['lady_pos_y']
            observation[11] = input_data['lady_pos_z']

            observation[12] = input_data['lady_vel_x']
            observation[13] = input_data['lady_vel_y']
            observation[14] = input_data['lady_vel_z']

            observation[15] = input_data['guard_pos_x']
            observation[16] = input_data['guard_pos_y']
            observation[17] = input_data['guard_pos_z']

            observation[18] = input_data['guard_vel_x']
            observation[19] = input_data['guard_vel_y']
            observation[20] = input_data['guard_vel_z']

            sun_position = None

        else:
            observation = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            observation[0] = input_data["time"]
            observation[1] = input_data["vehicle_mass"]
            observation[2] = input_data["vehicle_propellant"]

            observation[3] = input_data['pursuer_pos_x']
            observation[4] = input_data['pursuer_pos_y']
            observation[5] = input_data['pursuer_pos_z']

            observation[6] = input_data['pursuer_vel_x']
            observation[7] = input_data['pursuer_vel_y']
            observation[8] = input_data['pursuer_vel_z']

            observation[9] = input_data['evader_pos_x']
            observation[10] = input_data['evader_pos_y']
            observation[11] = input_data['evader_pos_z']

            observation[12] = input_data['evader_vel_x']
            observation[13] = input_data['evader_vel_y']
            observation[14] = input_data['evader_vel_z']

            sun_position = None
            if "sun_pos_x" in input_data:
                sun_position = [input_data['sun_pos_x'], input_data['sun_pos_y'], input_data['sun_pos_z']]

        vessel_up = None
        if "vessel_up_x" in input_data:
            vessel_up = np.array([input_data['vessel_up_x'], input_data['vessel_up_y'], input_data['vessel_up_z']])

        state = State(observation, vessel_up, sun_position)
        action = Action(ast.literal_eval(row['throttles']))

        """ Add state/action pair to sliding window.
        """
        sliding_window.add(state, action)

        # Skip null actions ALL except maybe the first one in a sequence of null actions
        if row['throttles'] == '[0, 0, 0]':
            if skip_all_null_actions:
                """ Skip all null actions
                """
                continue
            else:
                """ Skip all null actions except the first one in a sequence of consecutive
                null actions.
                """
                if skip_null_actions:
                    continue
                # Uncomment this line to skip all null actions after the first one in a sequence of consecutive null actions
                # skip_null_actions = True
        else:
            skip_null_actions = False

        """ Add messages to json list
        """
        messages = sliding_window.get_messages()
        message_structure = {
            'messages': messages
        }
        json_list.append(message_structure)

    # Save JSON to a file
    json_file_path = csv_file_path.replace('.csv', '.json')
    with open(json_file_path, 'w') as train:
        json.dump(json_list, train, indent=4)
    jsonl_file_path = json_file_path.replace('.json', '.jsonl')
    json_to_jsonl(json_file_path, jsonl_file_path)
    print(f"JSONL file saved: {jsonl_file_path}")


if __name__ == '__main__':
    # Load configuration from .env
    dotenv_path = join(dirname(__file__), '..', 'agents', '.env')
    load_dotenv(dotenv_path)

    use_relative_coordinates = State.DEFAULT_USE_RELATIVE_COORDINATES
    if 'USE_RELATIVE_COORDINATES' in os.environ:
        use_relative_coordinates = (os.environ['USE_RELATIVE_COORDINATES'].lower() == "true")

    use_short_names = State.DEFAULT_USE_SHORT_NAMES
    if 'USE_SHORT_NAMES' in os.environ:
        use_short_names = (os.environ['USE_SHORT_NAMES'].lower() == "true")

    use_enum = Action.DEFAULT_USE_ENUM
    if 'USE_ENUM' in os.environ:
        use_enum = (os.environ['USE_ENUM'].lower() == "true")

    use_prograde = State.DEFAULT_USE_PROGRADE
    if 'USE_PROGRADE' in os.environ:
        use_prograde = (os.environ['USE_PROGRADE'].lower() == "true")

    use_cot = State.DEFAULT_USE_COT
    if 'USE_COT' in os.environ:
        use_cot = (os.environ['USE_COT'].lower() == "true")

    use_cot_speed_limit = State.DEFAULT_USE_COT_SPEED_LIMIT
    if 'USE_COT_SPEED_LIMIT' in os.environ:
        use_cot_speed_limit = (os.environ['USE_COT_SPEED_LIMIT'].lower() == "true")

    sliding_window_size = SlidingWindow.DEFAULT_SLIDING_WINDOW_SIZE
    if 'SLIDING_WINDOW_SIZE' in os.environ:
        sliding_window_size = int(os.environ["SLIDING_WINDOW_SIZE"])

    sliding_window_stride = DEFAULT_SLIDING_WINDOW_STRIDE
    if 'SLIDING_WINDOW_STRIDE' in os.environ:
        sliding_window_stride = int(os.environ["SLIDING_WINDOW_STRIDE"])

    embed_history = SlidingWindow.DEFAULT_EMBED_HISTORY
    if 'EMBED_HISTORY' in os.environ:
        embed_history = (os.environ['EMBED_HISTORY'].lower() == "true")

    skip_all_null_actions = DEFAULT_SKIP_ALL_NULL_ACTIONS
    if 'SKIP_ALL_NULL_ACTIONS' in os.environ:
        skip_all_null_actions = (os.environ['SKIP_ALL_NULL_ACTIONS'].lower() == "true")

    # Process all CSV files in the current directory
    problem_env = input("Enter the problem and environment (e.g., pe1_i3): ")
    directory = '.'  # Current directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv") and filename.startswith(problem_env):
            csv_to_json_for_history(os.path.join(directory, filename), problem_env,
                                    use_relative_coordinates=use_relative_coordinates,
                                    use_short_names=use_short_names,
                                    use_enum=use_enum,
                                    use_prograde=use_prograde,
                                    use_cot=use_cot,
                                    use_cot_speed_limit=use_cot_speed_limit,
                                    size=sliding_window_size,
                                    stride=sliding_window_stride,
                                    embed_history=embed_history,
                                    skip_all_null_actions=skip_all_null_actions)
