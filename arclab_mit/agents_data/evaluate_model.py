import pandas as pd
import csv

import os
import sys
from os.path import join, dirname
from dotenv import load_dotenv

import numpy as np

from arclab_mit.agents.fine_tuning_agent import LLMAgent

if __name__== '__main__':

    if len (sys.argv) < 2:
        print ("Use: python evaluate_model.py <csv_file>")
        sys.exit(1)
    csv_filename = sys.argv[1]

    # Load configuration from .env
    dotenv_path = join(dirname(__file__), '..\\agents\.env')
    load_dotenv(dotenv_path)

    # Create the agent
    agent = LLMAgent()
    scenario = os.environ['SCENARIO']

    # Read the CSV file
    df = pd.read_csv(csv_filename)
    out_df = pd.DataFrame()
    for _, row in df.iterrows():
        input_data = {k: v for k, v in row.items() if k != 'throttles' and k != 'next_throttles'}

        observation = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        observation[0] = 0.0
        observation[1] = input_data["vehicle_mass"]
        observation[2] = input_data["vehicle_propellant"]

        observation[3] = input_data["pursuer_pos_x"]
        observation[4] = input_data["pursuer_pos_y"]
        observation[5] = input_data["pursuer_pos_z"]

        observation[6] = input_data["pursuer_vel_x"]
        observation[7] = input_data["pursuer_vel_y"]
        observation[8] = input_data["pursuer_vel_z"]

        observation[9] = input_data["evader_pos_x"]
        observation[10] = input_data["evader_pos_y"]
        observation[11] = input_data["evader_pos_z"]

        observation[12] = input_data["evader_vel_x"]
        observation[13] = input_data["evader_vel_y"]
        observation[14] = input_data["evader_vel_z"]

        if scenario.startswith('sb'):
            sun_position = np.array([input_data["sun_pos_x"], input_data["sun_pos_y"], input_data["sun_pos_z"]])

        # Get action
        action = agent.get_action(observation, sun_position)
        output_data = row
        output_data['action'] = action[0:3]
        out_df.append(output_data, ignore_index=True)

    out_filename = "fine_tuning/out_" + csv_filename
    # Writing to CSV file
    with open(out_filename, 'w', newline='') as csvfile:
        # Create a CSV writer object
        csv_writer = csv.writer(csvfile)

        # Write the data to the CSV file
        csv_writer.writerows(out_df)
