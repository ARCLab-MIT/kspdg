import pandas as pd
import json
import os
import numpy as np
import ast
from os.path import join, dirname
from dotenv import load_dotenv
from arclab_mit.agents.sliding_window import SlidingWindow
from arclab_mit.agents.agent_common import State, Action
import re

from datetime import datetime

dotenv_path = os.path.join(os.path.dirname(__file__), '..', 'agents', '.env')
load_dotenv(dotenv_path)

# Load configuration from alex_prompts.txt
dotenv_path = join(dirname(__file__), '..', 'agents', 'alex_prompts.txt')
load_dotenv(dotenv_path)

pattern = r"pe\d+_i\d+_keyboard_agent_actions_\d{8}-\d{6}\.csv"
# csv_file_path = r".\arclab_mit\agents_data\""
# csv_file_path = r".\"

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
                            skip_all_null_actions: bool = DEFAULT_SKIP_ALL_NULL_ACTIONS,
                            llama_format = None,
                            output_folder: str = None):
    df = pd.read_csv(csv_file_path)
    problem = problem_env[0:2].upper()
    sliding_window = SlidingWindow(size, problem_env.upper(),
                                   use_relative_coordinates, use_short_names, use_enum,
                                   use_prograde, use_cot, use_cot_speed_limit, embed_history,
                                   os.environ.get(f"{problem}_SYSTEM_PROMPT", ""),
                                   os.environ.get(f"{problem}_USER_PROMPT", ""),
                                   os.environ.get(f"{problem}_CHAIN_OF_THOUGHT", ""),
                                   os.environ.get("ASSISTANT_CONTENT", ""),
                                   os.environ.get(f"{problem}_HISTORY_PROMPT", ""),
                                   os.environ.get(f"{problem}_HISTORY_ITEM_PROMPT", ""))

    json_list = []
    alpaca_json_list = []
    sharegpt_json_list = []
    skip_null_actions = False

    for _, row in df.iterrows():
        input_data = {k: v for k, v in row.items() if k != 'throttles' and k != 'next_throttles'}

        if problem_env.lower().startswith('lbg'):
            observation = [input_data["time"], input_data["bandit_mass"], input_data["bandit_propellant"],
                           input_data['bandit_pos_x'], input_data['bandit_pos_y'], input_data['bandit_pos_z'],
                           input_data['bandit_vel_x'], input_data['bandit_vel_y'], input_data['bandit_vel_z'],
                           input_data['lady_pos_x'], input_data['lady_pos_y'], input_data['lady_pos_z'],
                           input_data['lady_vel_x'], input_data['lady_vel_y'], input_data['lady_vel_z'],
                           input_data['guard_pos_x'], input_data['guard_pos_y'], input_data['guard_pos_z'],
                           input_data['guard_vel_x'], input_data['guard_vel_y'], input_data['guard_vel_z']]
            sun_position = None
        else:
            observation = [input_data["time"], input_data["vehicle_mass"], input_data["vehicle_propellant"],
                           input_data['pursuer_pos_x'], input_data['pursuer_pos_y'], input_data['pursuer_pos_z'],
                           input_data['pursuer_vel_x'], input_data['pursuer_vel_y'], input_data['pursuer_vel_z'],
                           input_data['evader_pos_x'], input_data['evader_pos_y'], input_data['evader_pos_z'],
                           input_data['evader_vel_x'], input_data['evader_vel_y'], input_data['evader_vel_z']]
            sun_position = [input_data['sun_pos_x'], input_data['sun_pos_y'],
                            input_data['sun_pos_z']] if "sun_pos_x" in input_data else None

        vessel_up = np.array([input_data['vessel_up_x'], input_data['vessel_up_y'],
                              input_data['vessel_up_z']]) if "vessel_up_x" in input_data else None

        state = State(observation, vessel_up, sun_position)
        action = Action(ast.literal_eval(row['throttles']))

        sliding_window.add(state, action)

        if row['throttles'] == '[0, 0, 0]':
            if skip_all_null_actions:
                continue
            else:
                if skip_null_actions:
                    continue
                skip_null_actions = True
        else:
            skip_null_actions = False

        messages = sliding_window.get_messages()
        if llama_format is not None:
            system_prompt = ""
            user_msg_list = []
            model_answer_list = []
            for message in messages:
                if message['role'] == "system":
                    system_prompt = message['content']
                elif message['role'] == "user":
                    user_msg_list.append(message['content'])
                elif message['role'] == "assistant":
                    model_answer_list.append(message['content'])
            n = len(user_msg_list)

            llama_text = ""
            for i in range(n):
                text = llama_format.format(system_prompt=system_prompt,
                                                      user_msg=user_msg_list[i],
                                                      model_answer=model_answer_list[i])
                if i > 0:
                    text = re.sub("<<SYS>>.*<</SYS>>", "", text, flags=re.DOTALL)
                llama_text += text
            json_list.append({'text': llama_text})

            history = []
            for i in range(n-1):
                history.append([user_msg_list[i], model_answer_list[i]])
            alpaca_json_list.append({"instruction": user_msg_list[-1],
                                     "output": model_answer_list[-1],
                                     "system": system_prompt,
                                     "history": history})

            conversations = []
            for i in range(n):
                conversations.append({"from": "human",
                                      "value": user_msg_list[i]})
                conversations.append({"from": "gpt",
                                      "value": model_answer_list[i]})
            sharegpt_json_list.append({"conversations": conversations,
                "system": system_prompt})
        else:
            json_list.append({'messages': messages})

    if not output_folder:
        current_date = datetime.now().strftime("%m-%d-%y")
        counter = 1
        output_folder = f"training_ready_data_{current_date}-{counter}"
        while os.path.exists(output_folder):
            counter += 1
            output_folder = f"training_ready_data_{current_date}-{counter}"
        os.makedirs(output_folder)

    json_file_path = os.path.join(output_folder, os.path.basename(csv_file_path).replace('.csv', '.json'))
    with open(json_file_path, 'w') as train:
        json.dump(json_list, train, indent=4)

    jsonl_file_path = json_file_path.replace('.json', '.jsonl')
    json_to_jsonl(json_file_path, jsonl_file_path)
    print(f"JSONL file saved: {jsonl_file_path}")

    if llama_format is not None:
        # Save alpaca format
        alpaca_jsonl_file_path = json_file_path.replace('.json', '_alpaca.json')
        with open(alpaca_jsonl_file_path, 'w') as train:
            json.dump(alpaca_json_list, train, indent=4)

        # Save sharegpt format
        sharegpt_jsonl_file_path = json_file_path.replace('.json', '_sharegpt.json')
        with open(sharegpt_jsonl_file_path, 'w') as train:
            json.dump(sharegpt_json_list, train, indent=4)


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

    llama_format = None
    if 'LLAMA_FORMAT' in os.environ:
        llama_format = os.environ['LLAMA_FORMAT']

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
                                    skip_all_null_actions=skip_all_null_actions,
                                    llama_format=llama_format)
