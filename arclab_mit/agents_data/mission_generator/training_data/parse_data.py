from dotenv import load_dotenv
from os.path import join, dirname
from arclab_mit.agents.agent_common import State, Action
from arclab_mit.agents.sliding_window import SlidingWindow

import os
import glob
from datetime import datetime
from arclab_mit.agents_data import CSV_parsing

DEFAULT_SKIP_ALL_NULL_ACTIONS = False
DEFAULT_SLIDING_WINDOW_STRIDE = False

if __name__ == "__main__":
    # Load environment variables
    current_dir = os.path.dirname(__file__)

    # Join the current directory with the relative path to the .env file
    dotenv_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'agents', '.env'))

    print(dotenv_path)
    load_dotenv(dotenv_path)

    print(os.environ)

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

    llama_format = True
    if 'LLAMA_FORMAT' in os.environ:
        llama_format = os.environ['LLAMA_FORMAT']

    directory = './logs_05-19-2024/'

    problem_env = 'pe'
    # Get list of all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory, '*.csv'))

    # Create the folder with current date and counter
    current_date = datetime.now().strftime("%m-%d-%y")
    counter = 1
    output_folder = f"training_ready_data_{current_date}-{counter}"

    while os.path.exists(output_folder):
        counter += 1
        output_folder = f"training_ready_data_{current_date}_{counter}"
    os.makedirs(output_folder)

    print("Output folder: ", output_folder)

    # Process each CSV file
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        CSV_parsing.csv_to_json_for_history(csv_file, problem_env,
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
                                            llama_format=llama_format,
                                            output_folder=output_folder)
