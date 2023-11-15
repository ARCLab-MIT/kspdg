import json
import os


def load_and_parse_json_history(directory_path : str, problem_env : str):
    history = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".json") and filename.startswith('history_' + problem_env):
            file_path = os.path.join(directory_path, filename)
            # Load JSON file
            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    # Append the contents of this file to the history
                    history.extend(data)
            except json.JSONDecodeError:
                print(f"Error parsing JSON file: {file_path}")
    result = [item for sublist in history for item in sublist['messages']]
    return result

def get_history(problem_env: str = ''):
    if not problem_env:
        problem_env = input("Enter the problem and environment (e.g., pe1_i3): ")
    directory = '..\\agents_data\\'  # Current directory
    result = load_and_parse_json_history(os.path.join(directory), problem_env)
    return result
if __name__== '__main__':
    # Process all CSV files in the current directory
    assert(get_history() is not None)