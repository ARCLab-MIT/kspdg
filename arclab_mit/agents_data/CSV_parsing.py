import pandas as pd
import json
import os

import ast

pattern = r"pe\d+_i\d+_keyboard_agent_actions_\d{8}-\d{6}\.csv"
csv_file_path = r".\arclab_mit\agents_data\""

def csv_to_json_for_history(csv_file_path : str, problem_env : str):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Transform each row into the required JSON structure
    json_list = []
    for _, row in df.iterrows():
        input_data = {k: v for k, v in row.items() if k != 'throttles'}
        output_label = row['throttles']
        output_label = ast.literal_eval(output_label)

        message_structure = {
            "messages": [
                {"role": "user", "content": json.dumps(input_data)},
                {"role": "assistant", "content": "To solve the pursuer evader problem, we need to calculate the required throttles for the spacecraft. I will use the given parameters", "function_call": {"name" : "perform_action", "arguments": "{\n  \"forward_throttle\": " + str(output_label[0]) + ",\n  \"right_throttle\": " + str(output_label[1]) + ",\n  \"down_throttle\": " + str(output_label[2]) + "\n}"}}
            ]
        }
        json_list.append(message_structure)

    # Save JSON to a file
    json_file_path = csv_file_path.replace('.csv', '.json')
    json_file_path = json_file_path.replace(problem_env, 'history_' + problem_env)
    with open(json_file_path, 'w') as file:
        json.dump(json_list, file, indent=4)

    print(f"JSON file saved: {json_file_path}")

if __name__== '__main__':
    # Process all CSV files in the current directory
    problem_env = input("Enter the problem and environment (e.g., pe1_i3): ")
    directory = '.'  # Current directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv") and filename.startswith(problem_env):
            csv_to_json_for_history(os.path.join(directory, filename), problem_env)