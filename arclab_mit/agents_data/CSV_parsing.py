import pandas as pd
import json
import os

pattern = r"pe\d+_i\d+_keyboard_agent_actions_\d{8}-\d{6}\.csv"
csv_file_path = r".\arclab_mit\agents_data\""

def csv_to_json(csv_file_path):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Transform each row into the required JSON structure
    json_list = []
    for _, row in df.iterrows():
        input_data = {k: v for k, v in row.items() if k != 'throttles'}
        output_label = row['throttles']

        message_structure = {
            "messages": [
                {"role": "user", "content": json.dumps(input_data)},
                {"role": "assistant", "content": output_label}
            ]
        }
        json_list.append(message_structure)

    # Save JSON to a file
    json_file_path = csv_file_path.replace('.csv', '.json')
    with open(json_file_path, 'w') as file:
        json.dump(json_list, file, indent=4)

    print(f"JSON file saved: {json_file_path}")

if __name__== '__main__':
    # Process all CSV files in the current directory
    problem_env = input("Enter the problem and environment (e.g., pe1_i3): ")
    directory = '.'  # Current directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv") and filename.startswith(problem_env):
            csv_to_json(os.path.join(directory, filename))