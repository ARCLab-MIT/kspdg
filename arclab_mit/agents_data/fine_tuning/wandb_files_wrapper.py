"""
This script is a set of utils to publish and manage the results of evaluation runs in WANDB

"""

import json
import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv
from pylab import *
import seaborn as sns

from arclab_mit.agents.agent_common import State

from kspdg.pe1.pe1_base import PursuitEvadeGroup1Env


def export_run(run_id):
    """ Downloads all run files from WANDB and saves them in a local directory named
        as the basename of run_id.

        Args:
        run_id: full path to run id including entity and project
    """
    try:
        api = wandb.Api()
        run = api.run(run_id)
        basename = os.path.basename(run_id)
        os.mkdir(basename)
        os.chdir(basename)
        for file in run.files():
            file.download()
            print("Downloaded: " + file.name)
        with open("config.json", "w") as config_file:
            json.dump(run.config, config_file)

    except Exception as e:
        print ("Exception: " + str(e))


def import_run(entity, project, run_id, experiment):
    """ Uploads to WANDB all run files from local directory run_id. The new run
        will use the same 'entity', 'project' and 'experiment'.

        Args:
        entity: entity name
        project: project name
        run_id: run id
    """
    try:
        if os.path.exists(run_id):
            os.chdir(run_id)
            config = json.load(open("config.json"))
            wandb.init(
                # Set the entity where this run will be logged
                entity=entity,
                # Set the project where this run will be logged
                project=project,
                # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
                name=experiment,
                job_type = "LLM_fine_tune",
                # Track hyperparameters and run metadata
                config=config,
                )

            # Load metrics from the csv file
            history = pd.read_csv(join(config['job_id'], 'dataset', 'result_files.csv'))
            n_epochs = config['epochs']
            n_steps = len(history) - 1
            n_steps_per_epoch = int(n_steps / n_epochs)
            i = 0
            for epoch in range(n_epochs):
                for step in range(n_steps_per_epoch):
                    metrics = {'epoch': epoch,
                               'step': i,
                               'training_loss': float(history['train_loss'][i]),
                               'training_token_accuracy': float(history['train_accuracy'][i])}
                    if not pd.isna(history['valid_loss'][i]):
                        val_metrics = {
                            'validation_loss': float(history['valid_loss'][i]),
                            'validation_token_accuracy': float(history['valid_mean_token_accuracy'][i])}
                        metrics.update(val_metrics)
                    wandb.log({**metrics})
                    i += 1

        # Upload files to active run
        api = wandb.Api()
        new_run_id = entity + "/" + project + "/" + wandb.run.id
        run = api.run(new_run_id)
        for path, subdirs, files in os.walk("."):
            for name in files:
                full_path = join(path, name)
                if path.startswith(join(".", "ftjob")) or path.startswith(join(".", "results")):
                    run.upload_file(full_path)
                    print("Saved: " + full_path)

        # Mark the run as finished
        wandb.finish()

        print("Created run: " + new_run_id)

    except Exception as e:
        print ("Exception: " + str(e))


def upload_files(working_dir, path, run_id):
    """ Upload all files in local directory <working_dir>/<path> to an existing
        WANDB run_id.

        Args:
        working_dir: working directory for path
        path: relative poth to local file or directory
        run_id: full path to run id including entity and project
    """
    try:
        os.chdir(working_dir)
        api = wandb.Api()
        run = api.run(run_id)
        if os.path.isfile(path):
            filenames = [path]
        else:
            filenames = os.listdir(path)
            filenames = [join(path, f) for f in filenames]
        for filename in filenames:
            result = run.upload_file(filename)
            print("Saved: " + filename)

    except Exception as e:
        print ("Exception: " + str(e))


def generate_statistics_experiment(working_dir, experiment, extended_graphs=False):
    """ Collects data and save charts of the evaluations found in <working_dir>/<experiment>.

        Data is collected from:
        -   Agent logs (CSV and JSONL files).
        -   KSPDG result files with prefix 'kspdg_results'.

        Collected data:
        -   latency
        -   failure rate
        -   distance
        -   weighted score

        Charts are saved in the local directory in files with prefix 'fig_'. They include:
        -   latency (whisker chart)
        -   failure rate (bar chart)
        -   closest distance (whisker chart)
        -   weighted score (whisker chart)
        -   distance time series (line chart)
        -   distance and score time series (line chart)

        Args:
        working_dir: working directory
        experiment: relative poth to local file or directory
    """
    print("Generating statistics for experiment: " + experiment)
    try:
        """ Obtain list of files to be processed.
        """
        filenames = []
        csv_filenames = []
        result_filenames = []
        path = join(working_dir, experiment)
        os.chdir(path)
        for item in os.listdir(path):
            if item.endswith(".jsonl"):
                filenames.append(join(path, item))
        for item in os.listdir(path):
            if item.endswith(".csv"):
                csv_filenames.append(join(path, item))
        for item in os.listdir(path):
            if item.startswith("kspdg_results") and item.endswith(".txt"):
                result_filenames.append(join(path, item))

        df_series_dist = {}
        df_series_ws = {}

        """ Collect latency and failure from agent logs (JSONL files)
        """
        d = {'latency': [], 'failure': []}
        for filename in filenames:
            for line in open(filename, 'r'):
                data = json.loads(line)
                latency = data['end_time_ms'] - data['start_time_ms']
                d['latency'].append(latency)
                try:
                    failed = False
                    # Check wrong function call
                    if 'function_call' in data['outputs']['choices'][0]['message']:
                        d['failure'].append(data['outputs']['choices'][0]['message']['function_call']['name']
                                            not in ['perform_action'])
                    else:
                        content = data['outputs']['choices'][0]['message']['content']
                        function_name = "perform_action"
                        index = content.find(function_name + "(")
                        d['failure'].append(index == -1)
                except Exception as e:
                    d['failure'].append(True)
        df = pd.DataFrame(d, columns=['latency', 'failure'])

        """ Collect distance, weighted score and failures from agent logs (CSV files)
        """
        failure_data = []
        pe1Env = PursuitEvadeGroup1Env("pe1_i1_init")
        for filename in csv_filenames:
            with open(filename) as f:
                # Read first line stripping newlines
                header_line = f.readline()[:-1]
                header_columns = header_line.split(',')
            df_csv = pd.read_csv(filename, skiprows=1, header=None)
            if len(df_csv.columns) > len(header_columns):
                # There is a mistmach in columns. Extend with empty fields to match
                n = len(df_csv.columns) - len(header_columns)
                for i in range(0, n):
                    header_columns.append(f'tmp{n}')
            df_csv.columns = header_columns

            """ Collect weighted score 
            """
            # Ignore existing weighted score since we recalculate it
            if 'weighted_score' in df_csv.columns:
                # Existing column with weighted score is ignored
                del df_csv["weighted_score"]
            # Add weighted score column
            if 'weighted_score' not in df_csv.columns and experiment.lower().startswith('pe'):
                data = []
                closest_distance = sys.float_info.max
                init_mass = None
                for index, input_data in df_csv.iterrows():
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

                    state = State(observation, None)
                    if init_mass is None:
                        init_mass = state.vehicle_mass

                    if state.distance < closest_distance:
                        closest_distance = state.distance
                        weighted_score = pe1Env.get_weighted_score(state.distance, state.velocity, state.mission_time, init_mass - state.vehicle_mass)
                    data.append(weighted_score)
                df_csv['weighted_score'] = data

            basename = os.path.basename(filename)
            df_series_ws[basename] = df_csv

            """ Plot weighted score time series chart (line chart)
            """
            if 'weighted_score' in df_csv.columns:
                figure = plt.figure()
                best_score = df_csv.min()['weighted_score']
                line_chart = plt.plot(df_csv['time'], df_csv['weighted_score'], label=f'Best: {best_score:.2f}')
                plt.legend(loc="upper right")
                plt.xlabel('Time (s)')
                plt.ylabel('Score')
                plt.title(basename)
                figure.savefig("fig_ws_series_" + os.path.basename(filename) + ".png", format='png')
                plt.close(figure)

            """ Collect distance information
            """
            dist_data = []
            closest_dist_data = []
            closest_distance = sys.float_info.max
            pursuer_pos_data = []
            evader_pos_data = []
            rel_pos_data = []
            rel_vel_data = []
            vel_data = []
            for index, row in df_csv.iterrows():
                rel_pos = [row['evader_pos_x']-row['pursuer_pos_x'],
                           row['evader_pos_y']-row['pursuer_pos_y'],
                           row['evader_pos_z']-row['pursuer_pos_z']]
                rel_vel = [row['evader_vel_x']-row['pursuer_vel_x'],
                           row['evader_vel_y']-row['pursuer_vel_y'],
                           row['evader_vel_z']-row['pursuer_vel_z']]
                pursuer_pos = [row['pursuer_pos_x'],
                               row['pursuer_pos_y'],
                               row['pursuer_pos_z']]
                evader_pos = [row['evader_pos_x'],
                              row['evader_pos_y'],
                              row['evader_pos_z']]
                distance = np.linalg.norm(rel_pos, ord=2)
                velocity = np.linalg.norm(rel_vel, ord=2)
                if distance < closest_distance:
                    closest_distance = distance
                dist_data.append(distance)
                closest_dist_data.append(closest_distance)
                pursuer_pos_data.append(pursuer_pos)
                evader_pos_data.append(evader_pos)
                rel_pos_data.append(rel_pos)
                rel_vel_data.append(rel_vel)
                vel_data.append(velocity)
            df_csv['distance'] = dist_data
            df_csv['closest_distance'] = closest_dist_data
            df_csv['pursuer_pos'] = pursuer_pos_data
            df_csv['evader_pos'] = evader_pos_data
            df_csv['rel_pos'] = rel_pos_data
            df_csv['rel_vel'] = rel_vel_data
            df_csv['velocity'] = vel_data

            basename = os.path.basename(filename)
            df_series_dist[basename] = df_csv

            """ Plot distance time series chart (line chart)
            """
            figure = plt.figure()
            best_distance = df_csv.min()['distance']
            line_chart = plt.plot(df_csv['time'], df_csv['distance'], label=f'Best: {best_distance:.2f} m')
            plt.legend(loc="upper right")
            plt.xlabel('Time (s)')
            plt.ylabel('Distance (m)')
            plt.title(basename)
            figure.savefig("fig_dist_series_" + os.path.basename(filename) + ".png", format='png')
            plt.close(figure)

            if extended_graphs:
                """ Plot celestial body relative position time series chart (multiple plot line chart)
                """
                best_distance = df_csv.min()['distance']
                figure, axs = plt.subplots(4, constrained_layout=True)

                axs[0].tick_params(axis='x', which='major', labelsize=8)
                axs[0].tick_params(axis='y', which='major', labelsize=8)
                l1, = axs[0].plot(df_csv['time'],  [p[0] for p in df_csv['rel_pos']], c='blue', label='position')
                axs[0].set_xlabel('Time (s)', fontsize=8)
                axs[0].set_ylabel(r'$\Delta$x (m)', fontsize=8)
    #            axs[0].set_yscale('symlog')
                axs[0].grid(linestyle='--')

                axs[1].tick_params(axis='x', which='major', labelsize=8)
                axs[1].tick_params(axis='y', which='major', labelsize=8)
                l1, = axs[1].plot(df_csv['time'],  [p[1] for p in df_csv['rel_pos']], c='blue', label='position')
                axs[1].set_xlabel('Time (s)', fontsize=8)
                axs[1].set_ylabel(r'$\Delta$y (m)', fontsize=8)
    #            axs[1].set_yscale('symlog')
                axs[1].grid(linestyle='--')

                axs[2].tick_params(axis='x', which='major', labelsize=8)
                axs[2].tick_params(axis='y', which='major', labelsize=8)
                l1, = axs[2].plot(df_csv['time'],  [p[2] for p in df_csv['rel_pos']], c='blue', label='position')
                axs[2].set_xlabel('Time (s)', fontsize=8)
                axs[2].set_ylabel(r'$\Delta$z (m)', fontsize=8)
    #            axs[2].set_yscale('symlog')
                axs[2].grid(linestyle='--')

                axs[3].tick_params(axis='x', which='major', labelsize=8)
                axs[3].tick_params(axis='y', which='major', labelsize=8)
                l1, = axs[3].plot(df_csv['time'],  df_csv['distance'], c='green', label='position')
                axs[3].set_xlabel('Time (s)', fontsize=8)
                axs[3].set_ylabel('Distance (m)', fontsize=8)
                axs[3].grid(linestyle='--')

                figure.savefig("fig_rel_pos_series_" + os.path.basename(filename) + ".png", format='png')
                plt.close(figure)

                """ Plot celestial body relative position and velocity time series chart (multiple plot line chart)
                """
                best_distance = df_csv.min()['distance']
                figure, axs = plt.subplots(4, constrained_layout=True)

                axs[0].tick_params(axis='x', which='major', labelsize=8)
                axs[0].tick_params(axis='y', which='major', labelsize=8)
                l1, = axs[0].plot(df_csv['time'],  [p[0] for p in df_csv['rel_pos']], c='green', label='position')
                axs[0].set_xlabel('Time (s)', fontsize=8)
                axs[0].set_ylabel(r'$\Delta$x (m)', fontsize=8)
    #            axs[0].set_yscale('symlog')
                axs[0].grid(linestyle='--')

                axs_speed = axs[0].twinx()
                axs_speed.tick_params(axis='y', which='major', labelsize=8)
                l2, = axs_speed.plot(df_csv['time'],  [p[0] for p in df_csv['rel_vel']], c='blue', label='velocity')
                axs_speed.set_ylabel(r'$\Delta$vx (m/s)', fontsize=8)
                axs_speed.grid(linestyle='--')

                plt.legend([l1, l2], ["position", "velocity"], loc="upper right", fontsize=8)

                axs[1].tick_params(axis='x', which='major', labelsize=8)
                axs[1].tick_params(axis='y', which='major', labelsize=8)
                l1, = axs[1].plot(df_csv['time'],  [p[1] for p in df_csv['rel_pos']], c='green', label='position')
                axs[1].set_xlabel('Time (s)', fontsize=8)
                axs[1].set_ylabel(r'$\Delta$y (m)', fontsize=8)
    #            axs[1].set_yscale('symlog')
                axs[1].grid(linestyle='--')

                axs_speed = axs[1].twinx()
                axs_speed.tick_params(axis='y', which='major', labelsize=8)
                l2, = axs_speed.plot(df_csv['time'],  [p[1] for p in df_csv['rel_vel']], c='blue', label='velocity')
                axs_speed.set_ylabel(r'$\Delta$vy (m/s)', fontsize=8)
                axs_speed.grid(linestyle='--')

                plt.legend([l1, l2], ["position", "velocity"], loc="upper right", fontsize=8)

                axs[2].tick_params(axis='x', which='major', labelsize=8)
                axs[2].tick_params(axis='y', which='major', labelsize=8)
                l1, = axs[2].plot(df_csv['time'],  [p[2] for p in df_csv['rel_pos']], c='green', label='position')
                axs[2].set_xlabel('Time (s)', fontsize=8)
                axs[2].set_ylabel(r'$\Delta$z (m)', fontsize=8)
    #            axs[2].set_yscale('symlog')
                axs[2].grid(linestyle='--')

                axs_speed = axs[2].twinx()
                axs_speed.tick_params(axis='y', which='major', labelsize=8)
                l2, = axs_speed.plot(df_csv['time'],  [p[2] for p in df_csv['rel_vel']], c='blue', label='velocity')
                axs_speed.set_ylabel(r'$\Delta$vz (m/s)', fontsize=8)
                axs_speed.grid(linestyle='--')

                plt.legend([l1, l2], ["position", "velocity"], loc="upper right", fontsize=8)

                axs[3].tick_params(axis='x', which='major', labelsize=8)
                axs[3].tick_params(axis='y', which='major', labelsize=8)
                l1, = axs[3].plot(df_csv['time'],  df_csv['distance'], c='green', label='position')
                axs[3].set_xlabel('Time (s)', fontsize=8)
                axs[3].set_ylabel('Distance (m)', fontsize=8)
                axs[3].grid(linestyle='--')

                axs_speed = axs[3].twinx()
                axs_speed.tick_params(axis='y', which='major', labelsize=8)
                l2, = axs_speed.plot(df_csv['time'],  df_csv['velocity'], c='blue', label='speed')
                axs_speed.set_ylabel(r'Speed (m/s)', fontsize=8)
                axs_speed.grid(linestyle='--')

                plt.legend([l1, l2], ["distance", "speed"], loc="upper right", fontsize=8)

                figure.savefig("fig_rel_pos_vel_series_" + os.path.basename(filename) + ".png", format='png')
                plt.close(figure)

                """ Plot celestial body pursuer and evader position time series chart (multiple plot line chart)
                """
                best_distance = df_csv.min()['distance']
                figure, axs = plt.subplots(4, constrained_layout = True)
                axs[0].tick_params(axis='y', which='major', labelsize=8)
                axs[0].plot(df_csv['time'],  [p[0] for p in df_csv['pursuer_pos']], c='green')
                axs[0].plot(df_csv['time'],  [p[0] for p in df_csv['evader_pos']], c='red')
                axs[0].set_xlabel('Time (s)')
                axs[0].set_ylabel('x (m)')

                axs[1].tick_params(axis='y', which='major', labelsize=8)
                axs[1].plot(df_csv['time'],  [p[1] for p in df_csv['pursuer_pos']], c='green')
                axs[1].plot(df_csv['time'],  [p[1] for p in df_csv['evader_pos']], c='red')
                axs[1].set_xlabel('Time (s)')
                axs[1].set_ylabel('y (m)')

                axs[2].tick_params(axis='y', which='major', labelsize=8)
                axs[2].plot(df_csv['time'],  [p[2] for p in df_csv['pursuer_pos']], c='green')
                axs[2].plot(df_csv['time'],  [p[2] for p in df_csv['evader_pos']], c='red')
                axs[2].set_xlabel('Time (s)')
                axs[2].set_ylabel('z (m)')

                axs[3].tick_params(axis='y', which='major', labelsize=8)
                axs[3].plot(df_csv['time'],  df_csv['distance'], c='green')
                axs[3].set_xlabel('Time (s)')
                axs[3].set_ylabel('Distance (m)')

                figure.savefig("fig_pursuer_evader_series_" + os.path.basename(filename) + ".png", format='png')
                plt.close(figure)

                """ Plot celestial body relative position time series chart (3D Chart)
                """
                figure = plt.figure()
                ax = plt.axes(projection ='3d')
                ax.tick_params(axis='x', labelsize=6, rotation=45, pad=5)
                for label in ax.get_xticklabels():
                    label.set_horizontalalignment('center')
                    label.set_verticalalignment('bottom')
                ax.tick_params(axis='y', labelsize=6, rotation=-15, pad=5)
                for label in ax.get_yticklabels():
                    label.set_horizontalalignment('center')
                    label.set_verticalalignment('bottom')
                ax.tick_params(axis='z', labelsize=6, pad=0)

                ax.set_xlabel(r'$\Delta$x (m)')
                ax.set_ylabel(r'$\Delta$y (m)')
                ax.set_zlabel(r'$\Delta$z (m)')
    #            ax.invert_yaxis()
                """
                ax.set_xscale('symlog')
                ax.set_yscale('symlog')
                ax.set_zscale('symlog')
                """
                x = [p[0] for p in df_csv['rel_pos']]
                y = [p[1] for p in df_csv['rel_pos']]
                z = [p[2] for p in df_csv['rel_pos']]
                u = [p[0] for p in df_csv['rel_vel']]
                v = [p[1] for p in df_csv['rel_vel']]
                w = [p[2] for p in df_csv['rel_vel']]
                ax.plot3D(x, y, z, 'green')
                ax.scatter(x[-1],y[-1],z[-1], '-', c="green")
                ax.quiver(x, y, z, u, v, w, color="blue", arrow_length_ratio=0.02)
    #            ax.set_title('Relative position between pursuer and evader')
                figure.savefig("fig_3D_rel_pos_series_" + os.path.basename(filename) + ".png", format='png')
                plt.close(figure)

                """ Plot celestial body pursuer and evader positions time series chart (3D Chart)
                """
                figure = plt.figure()
                ax = plt.axes(projection='3d')
                ax.tick_params(axis='x', labelsize=6, rotation=45, pad=10)
                for label in ax.get_xticklabels():
                    label.set_horizontalalignment('center')
                    label.set_verticalalignment('bottom')
                ax.tick_params(axis='y', labelsize=6, rotation=-15, pad=5)
                for label in ax.get_yticklabels():
                    label.set_horizontalalignment('center')
                    label.set_verticalalignment('bottom')
                ax.tick_params(axis='z', labelsize=6, pad=0)

                ax.set_xlabel('x (m)')
                ax.set_ylabel('y (m)')
                ax.set_zlabel('z (m)')

    #            ax.invert_yaxis()

                x = [p[0] for p in df_csv['pursuer_pos']]
                y = [p[1] for p in df_csv['pursuer_pos']]
                z = [p[2] for p in df_csv['pursuer_pos']]
                ax.plot3D(x, y, z, 'green', label='pursuer')
                ax.scatter(x[0], y[0], z[0], '-', c="green")
                ax.scatter(x[-1],y[-1],z[-1], '-', c="green")
                x = [p[0] for p in df_csv['evader_pos']]
                y = [p[1] for p in df_csv['evader_pos']]
                z = [p[2] for p in df_csv['evader_pos']]
                ax.plot3D(x, y, z, 'red', label='evader')
                ax.scatter(x[0], y[0], z[0], '-', c="red")
                ax.scatter(x[-1], y[-1], z[-1], '-', c="red")
    #            ax.scatter(x, y, z, c="red")
    #
                elev = 4
                azim = -30
                roll = 0
                ax.view_init(elev, azim, roll)
                plt.legend(loc="upper right")
                figure.align_labels()
                figure.savefig("fig_3D_pursuer_evader_series_" + os.path.basename(filename) + ".png", format='png')
                # plt.show()
                plt.close(figure)

                """ Plot celestial body pursuer and evader positions time series chart (2D Chart)
                """
                figure = plt.figure()
                ax = plt.axes()
                ax.set_xlabel('x (m)')
                ax.set_ylabel('y (m)')
                ax.invert_yaxis()
                x = [p[0] for p in df_csv['pursuer_pos']]
                y = [p[1] for p in df_csv['pursuer_pos']]
                ax.plot(x, y, 'green', label='pursuer')
    #            ax.scatter(x[-1], y[-1], '-', c="green")
                x = [p[0] for p in df_csv['evader_pos']]
                y = [p[1] for p in df_csv['evader_pos']]
                ax.plot(x, y, 'red', label='evader')
    #            ax.scatter(x[-1], y[-1], '-', c="red")
                plt.legend(loc="upper right")
                figure.savefig("fig_2D_pursuer_evader_series_" + os.path.basename(filename) + ".png", format='png')
                plt.close(figure)

            """ Collect failures
            """
            for index, row in df_csv.iterrows():
                failure_data.append(row['duration'] == 0.1)

        # Add failure data to df
        df['failure_2'] = failure_data


        """ Collect distance and weighted score from KSPDG result files
        """
        d_results = {'closest_distance': [], 'weighted_score': [], 'closest_approach_speed': [], 'closest_approach_pursuer_fuel_usage': []}
        for filename in result_filenames:
            str = ""
            prev_line = None
            for line in open(filename, 'r'):
                if prev_line is not None:
                    str += prev_line
                prev_line = line
            data = json.loads(str)
            d_results['closest_distance'].append(data['agent_env_results']['closest_approach'])
            d_results['weighted_score'].append(data['agent_env_results']['weighted_score'])
            d_results['closest_approach_speed'].append(data['agent_env_results']['closest_approach_speed'])
            d_results['closest_approach_pursuer_fuel_usage'].append(data['agent_env_results']['closest_approach_pursuer_fuel_usage'])

        df_results = pd.DataFrame(d_results, columns=['closest_distance', 'weighted_score', 'closest_approach_speed', 'closest_approach_pursuer_fuel_usage'])

        """ Create summary of KSPDG result files
        """
        statistics_filename = "statistics.txt"
        with open(statistics_filename, "w") as file:
            file.write("Run statistics:\n" + df.describe(include='all').to_string())
            file.write("\n\nResults statistics:\n" + df_results.describe(include='all').to_string())

        """ Plot latency chart (whisker chart)
        """
        figure = plt.figure(figsize=(8,6))
        bplot_df = pd.DataFrame({experiment: df['latency']})
        bplot, bp_dict = bplot_df.boxplot(column=[experiment], vert=True, grid=False, showfliers=False, return_type='both')

        # Overlay values
        for line in bp_dict['medians']:
            # get position data for median line
            x, y = line.get_xydata()[1] # top of median line
            # overlay median value
            text(x, y, '%.1f' % y,
                 horizontalalignment='left', # left
                 verticalalignment='center')      # centered
        for line in bp_dict['boxes']:
            x, y = line.get_xydata()[0] # bottom of left line
            text(x,y, '%.1f' % y,
                 horizontalalignment='right', # right
                 verticalalignment='center')      # centered
            x, y = line.get_xydata()[3] # bottom of right line
            text(x,y, '%.1f' % y,
                 horizontalalignment='right', # right
                 verticalalignment='center')      # centered

        bplot.set_ylabel('Latency (ms)')
        figure.tight_layout()
        figure.savefig("fig_latency.png", format='png')
        plt.close(figure)

        """ Plot failure rate chart (bar chart)
        """
        figure = plt.figure(figsize=(8,6))
        frequencies = df['failure'].value_counts() / len(df) * 100
        frequencies = round(frequencies, 2)
        bar_chart = plt.bar(np.arange(len(frequencies)), frequencies)
        for rect1 in bar_chart:
            height = rect1.get_height()
            plt.annotate("{}%".format(height), (rect1.get_x() + rect1.get_width() / 2, height + .05), ha="center",
                         va="bottom", fontsize=15)
        plt.xticks([0, 1], ["Success", "Failure"])
        plt.ylabel('Frequency')
        figure.tight_layout()
        figure.savefig("fig_failure.png", format='png')
        plt.close(figure)

        """ Plot closest distance chart (whisker chart)
        """
        figure = plt.figure(figsize=(8,6))
        bplot_df = pd.DataFrame({experiment: df_results['closest_distance']})
        bplot, bp_dict = bplot_df.boxplot(column=[experiment], vert=True, grid=False, showfliers=False, return_type='both')

        # Overlay values
        for line in bp_dict['medians']:
            # get position data for median line
            x, y = line.get_xydata()[1] # top of median line
            # overlay median value
            text(x, y, '%.1f' % y,
                 horizontalalignment='left', # left
                 verticalalignment='center')      # centered
        for line in bp_dict['boxes']:
            x, y = line.get_xydata()[0] # bottom of left line
            text(x,y, '%.1f' % y,
                 horizontalalignment='right', # right
                 verticalalignment='center')      # centered
            x, y = line.get_xydata()[3] # bottom of right line
            text(x,y, '%.1f' % y,
                 horizontalalignment='right', # right
                 verticalalignment='center')      # centered

        bplot.set_ylabel('closest distance (m)')
        figure.tight_layout()
        figure.savefig("fig_closest_distance.png", format='png')
        plt.close(figure)

        """ Plot weighted score chart (whisker chart)
        """
        figure = plt.figure(figsize=(8,6))
        bplot_df = pd.DataFrame({experiment: df_results['weighted_score']})
        bplot, bp_dict = bplot_df.boxplot(column=[experiment], vert=True, grid=False, showfliers=False, return_type='both')

        # Overlay values
        for line in bp_dict['medians']:
            # get position data for median line
            x, y = line.get_xydata()[1] # top of median line
            # overlay median value
            text(x, y, '%.1f' % y,
                 horizontalalignment='left', # left
                 verticalalignment='center')      # centered
        for line in bp_dict['boxes']:
            x, y = line.get_xydata()[0] # bottom of left line
            text(x,y, '%.1f' % y,
                 horizontalalignment='right', # right
                 verticalalignment='center')      # centered
            x, y = line.get_xydata()[3] # bottom of right line
            text(x,y, '%.1f' % y,
                 horizontalalignment='right', # right
                 verticalalignment='center')      # centered

        bplot.set_ylabel('Score')
        plt.yscale("log")
        figure.tight_layout()
        figure.savefig("fig_weighted_score.png", format='png')
        plt.close(figure)

    except Exception as e:
        print ("Exception: " + str(e))
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    return df, df_results, df_series_dist, df_series_ws


def generate_statistics(working_dir, scenario):
    """ Collects data and save charts of the evaluations found in <working_dir> for <scenario>.

        Evaluations are found in local directories with prefix <working_dir>/<scenario> where
        each directory represents an experiment.

        First process each experiment individually. Then for a subset of experiments:
        -   Consolidates latency, distance and score information and saves it to 'summary.txt'.
        -   Plots the following charts:
            -   Latency per experiment (whisker chart)
            -   Failure rate per experiment (bar chart)
            -   Closest distance per experiment (whisker chart)
            -   Weighted score per experiment (whisker chart)
            -   Distance time series of best run per experiment (line chart)
            -   Weighted score time series of best run per experiment (line chart)
            -   Distance time series of best run among all experiments (line chart)

        Charts are saved in files with prefix 'fig_'.

        Args:
        working_dir: working directory.
        experiment: prefix of the experiment directories found in <working_dir>.
    """

    # Obtain experiment directories
    experiments = [os.path.basename(f.path) for f in os.scandir(working_dir)
                    if f.is_dir() and os.path.basename(f.path).startswith(scenario)]

    """ Experiment labels to use in charts
    """
    experiment_labels = {
        'PE1_LLM': 'Baseline LLM',
        'PE1_w_o_system_prompt_lrm_2': 'Simple fine tuning',
        'PE1_w_o_system_prompt_lrm_0.2': '+ hyperparameter tuning',
        'PE1_w_system_prompt_lrm_0.2': '+ system prompt',
        'PE1_w_system_prompt_multfiles_lrm_0.2': '+ two train gameplays',
        'PE1_GamePlay_1': 'Gameplay 1',
        'PE1_GamePlay_2': 'Gameplay 2',
        'PE1_test': 'PE1 test',
        'PE1_LLM_cot_navball': 'LLM with CoT navball',
        'PE1_LLM_0125': 'LLM 0125',
        'PE1_w_system_prompt_three_files_lrm_0.2': '+ three train gameplays',
        'PE1_E1_I3_LLM_0125_CoT_Navball': 'w/ CoT - Scenario E1',
        'PE1_E1_I3_LLM_0125_CoT_Navball_fix': 'w/ CoT - Scenario E1 (fix)',
        'PE1_E2_I3_LLM_0125_CoT_Navball': 'w/ CoT - Scenario E2',
        'PE1_E2_I3_LLM_0125_CoT_Navball_fix': 'w/ CoT - Scenario E2 (fix)',
        'PE1_E3_I3_LLM_0125_CoT_Navball': 'w/ CoT - Scenario E3',
        'PE1_E3_I3_LLM_0125_CoT_Navball_fix': 'w/ CoT (base model) - Scenario E3 (fix)',
        'PE1_E4_I3_LLM_0125_CoT_Navball': 'w/ CoT - Scenario E4',
        'PE1_E4_I3_LLM_0125_CoT_Navball_fix': 'w/ CoT - Scenario E4 (fix)',
        'PE1_E2_I4_LLM_0125_CoT_Navball': 'w/ CoT - Scenario E2_I4',
        'PE1_E3_I3_LLM_0125_CoT_Navball_speed_limit_20': 'w/ CoT & speed limit 20 m/s - Scenario E3',
        'PE1_E3_I3_LLM_0125_CoT_Navball_speed_limit_30': 'w/ CoT & speed limit 30 m/s - Scenario E3',
        'PE1_navball': 'Navball (bot) - Scenario E3',
        'PE1_E3_I3_LLM_0125_CoT_Navball_lrm_0.2': 'w/ CoT (fine tuned) - Scenario E3',
        'PE1_E3_I3_Llama_Agent_dur_1': 'Llama3 Base Model - Scenario E3',
        'PE1_E3_I3_Llama_Agent_FineTuning_25_dur_1': 'Llama3 Fine Tuned 25 files - Scenario E3',
        'PE1_E3_I3_Llama_Agent_FineTuning_25_dur_0.5': 'Llama3 Fine Tuned 25 files - Scenario E3 (dur=0.5)',
        'PE1_E3_I3_Llama_Agent_FineTuning_25_dur_2': 'Llama3 Fine Tuned 25 files - Scenario E3 (dur=2)',
        'PE1_E3_I3_Llama_Agent_FineTuning_25_dur_1_return_throttles': 'Llama3 Fine Tuned 25 files (optimized) - Scenario E3',
        'PE1_E3_I3_Llama_Agent_FineTuning_10_dur_1_return_throttles': 'Llama3 Fine Tuned 10 files (optimized) - Scenario E3',
        'PE1_E3_I3_Llama_Agent_FineTuning_50_dur_1_return_throttles': 'Llama3 Fine Tuned 50 files (optimized) - Scenario E3',
        'PE1_E3_I3_Llama_Agent_FineTuning_10_win_3_dur_1_return_throttles': 'Llama3 Fine Tuned 10 files win=3 (optimized) - Scenario E3',
        'PE1_E3_I3_Navball_delay_0_seconds': 'Navball (bot) - Scenario E3 (delay=0)',
    }

    """ Experiment directories to include for consolidation
    """
    include = ['PE1_LLM',
               'PE1_w_o_system_prompt_lrm_2',
               'PE1_w_o_system_prompt_lrm_0.2',
               'PE1_w_system_prompt_lrm_0.2',
               'PE1_w_system_prompt_multfiles_lrm_0.2']
    include = ['PE1_GamePlay_1',
               'PE1_GamePlay_2']
    include = ['PE1_E1_I3_LLM_0125_CoT_Navball',
               'PE1_E2_I3_LLM_0125_CoT_Navball',
               'PE1_E3_I3_LLM_0125_CoT_Navball',
               'PE1_E4_I3_LLM_0125_CoT_Navball']
    # include = ['PE1_w_system_prompt_three_files_lrm_0.2']
    include = ['PE1_navball',
               'PE1_E3_I3_LLM_0125_CoT_Navball',
               'PE1_E3_I3_LLM_0125_CoT_Navball_speed_limit_20',
               'PE1_E3_I3_LLM_0125_CoT_Navball_speed_limit_30']
    include = ['PE1_E1_I3_LLM_0125_CoT_Navball',
               'PE1_E1_I3_LLM_0125_CoT_Navball_fix',
               'PE1_E2_I3_LLM_0125_CoT_Navball',
               'PE1_E2_I3_LLM_0125_CoT_Navball_fix',
               'PE1_E3_I3_LLM_0125_CoT_Navball',
               'PE1_E3_I3_LLM_0125_CoT_Navball_fix',
               'PE1_E4_I3_LLM_0125_CoT_Navball',
               'PE1_E4_I3_LLM_0125_CoT_Navball_fix']
    include = ['PE1_E3_I3_LLM_0125_CoT_Navball_fix',
               'PE1_E3_I3_LLM_0125_CoT_Navball_lrm_0.2']
    include = ['PE1_E3_I3_Llama_Agent_dur_1',
               'PE1_E3_I3_Llama_Agent_FineTuning_25_dur_1',
               'PE1_E3_I3_Llama_Agent_FineTuning_25_dur_0.5',
               'PE1_E3_I3_Llama_Agent_FineTuning_25_dur_2']
    include = ['PE1_E3_I3_Llama_Agent_dur_1',
               'PE1_E3_I3_Llama_Agent_FineTuning_10_dur_1_return_throttles',
               'PE1_E3_I3_Llama_Agent_FineTuning_25_dur_1_return_throttles',
               'PE1_E3_I3_Llama_Agent_FineTuning_50_dur_1_return_throttles']
    include = ['PE1_E3_I3_Llama_Agent_FineTuning_10_dur_1_return_throttles',
               'PE1_E3_I3_Llama_Agent_FineTuning_10_win_3_dur_1_return_throttles']
    include = ['PE1_E3_I3_Llama_Agent_FineTuning_25_dur_1_return_throttles',
               'PE1_E3_I3_Navball_delay_0_seconds']
    include = ['PE1_E3_I3_Navball_delay_0_seconds',
               'PE1_E3_I3_Llama_Agent_FineTuning_10_dur_1_return_throttles',
               'PE1_E3_I3_Llama_Agent_FineTuning_25_dur_1_return_throttles',
               'PE1_E3_I3_Llama_Agent_FineTuning_50_dur_1_return_throttles']
    include = ['PE1_E3_I3_Navball_delay_0_seconds',
               'PE1_E3_I3_Llama_Agent_dur_1',
               'PE1_E3_I3_Llama_Agent_FineTuning_10_dur_1_return_throttles',
               'PE1_E3_I3_Llama_Agent_FineTuning_25_dur_1_return_throttles',
               'PE1_E3_I3_Llama_Agent_FineTuning_50_dur_1_return_throttles',
               'PE1_E3_I3_Llama_Agent_FineTuning_10_win_3_dur_1_return_throttles']

    """ Best runs for each experiment. They are identified by the filename of the
        agent log (CSV file) which resulted in lowest score.
        
        NOTE: KSDPG result file and agent logs use different timestamps in the filenames which
        makes it difficult to determine the agent logs for the best run. This could be improved
        in the future.
    """
    experiment_with_best_run = 'PE1_w_system_prompt_multfiles_lrm_0.2'
    best_runs = {
        'Baseline LLM': 'fine_tune_agent_log_PE1_E3_I3_20240216-153548.csv',
        'Simple fine tuning': 'fine_tune_agent_log_PE1_E3_I3_20240216-151534.csv',
        '+ hyperparameter tuning': 'fine_tune_agent_log_PE1_E3_I3_20231229-164511.csv',
        '+ system prompt': 'fine_tune_agent_log_PE1_E3_I3_20231229-205737.csv',
        '+ two train gameplays': 'fine_tune_agent_log_PE1_E3_I3_20240216-142257.csv',
        'PE1 test': 'fine_tune_agent_log_PE1_E3_I3_20240217-220530.csv',
        'LLM with CoT navball': 'fine_tune_agent_log_PE1_E2_I3_20240218-143227.csv',
        'LLM 0125': 'fine_tune_agent_log_PE1_E2_I3_20240220-112235.csv',
        '+ three train gameplays': 'fine_tune_agent_log_PE1_E3_I3_20240220-141012.csv',
        'w/ CoT - Scenario E1': 'fine_tune_agent_log_PE1_E3_I3_20240221-170359.csv',
        'w/ CoT - Scenario E1 (fix)': 'fine_tune_agent_log_PE1_E3_I3_20240225-183111.csv',
        'w/ CoT - Scenario E2': 'fine_tune_agent_log_PE1_E3_I3_20240221-175956.csv',
        'w/ CoT - Scenario E2 (fix)': 'fine_tune_agent_log_PE1_E3_I3_20240225-185118.csv',
        'w/ CoT - Scenario E3': 'fine_tune_agent_log_PE1_E3_I3_20240221-165645.csv',
#        'w/ CoT (base model) - Scenario E3 (fix)': 'fine_tune_agent_log_PE1_E3_I3_20240224-171056.csv',
        'w/ CoT (base model) - Scenario E3 (fix)': 'fine_tune_agent_log_PE1_E3_I3_20240224-172845.csv',
#        'w/ CoT (fine tuned) - Scenario E3': 'fine_tune_agent_log_PE1_E3_I3_20240302-140326.csv',
        'w/ CoT (fine tuned) - Scenario E3': 'fine_tune_agent_log_PE1_E3_I3_20240302-134925.csv',
        'w/ CoT - Scenario E4': 'fine_tune_agent_log_PE1_E3_I3_20240221-190242.csv',
        'w/ CoT - Scenario E4 (fix)': 'fine_tune_agent_log_PE1_E3_I3_20240224-202218.csv',
        'w/ CoT - Scenario E2_I4': 'fine_tune_agent_log_PE1_E3_I3_20240221-190938.csv',
        'w/ CoT & speed limit 20 m/s - Scenario E3': 'fine_tune_agent_log_PE1_E3_I3_20240224-140430.csv',
        'w/ CoT & speed limit 30 m/s - Scenario E3': 'fine_tune_agent_log_PE1_E3_I3_20240224-141137.csv',
        'Navball (bot) - Scenario E3': 'PE1_E3_I3_navball_log_20240215-115242.csv',
        'Llama3 Base Model - Scenario E3': 'llama_fine_tune_agent_log_PE1_E3_I3_20240616-104825.csv',
        'Llama3 Fine Tuned 25 files - Scenario E3': 'llama_fine_tune_agent_log_PE1_E3_I3_20240616-094012.csv',
        'Llama3 Fine Tuned 25 files - Scenario E3 (dur=0.5)': 'llama_fine_tune_agent_log_PE1_E3_I3_20240616-092943.csv',
        'Llama3 Fine Tuned 25 files - Scenario E3 (dur=2)': 'llama_fine_tune_agent_log_PE1_E3_I3_20240616-090506.csv',
        'Llama3 Fine Tuned 25 files (optimized) - Scenario E3': 'llama_fine_tune_agent_log_PE1_E3_I3_20240618-114801.csv',
        'Llama3 Fine Tuned 10 files (optimized) - Scenario E3': 'llama_fine_tune_agent_log_PE1_E3_I3_20240618-095800.csv',
        'Llama3 Fine Tuned 50 files (optimized) - Scenario E3': 'llama_fine_tune_agent_log_PE1_E3_I3_20240618-101213.csv',
        'Llama3 Fine Tuned 10 files win=3 (optimized) - Scenario E3': 'llama_fine_tune_agent_log_PE1_E3_I3_20240627-135210.csv',
        'Navball (bot) - Scenario E3 (delay=0)': 'navball_log_PE1_E3_I3_20240630-124618.csv',
    }

    """ Process experiment directories
    """
    df_dict = {}
    df_results_dict = {}
    df_series_dist_dict = {}
    df_series_ws_dict = {}
    for experiment in experiments:
        if experiment in include:
            df, df_results, df_series_dist, df_series_ws = generate_statistics_experiment(working_dir, experiment)
            df_dict[experiment_labels[experiment]] = df
            df_results_dict[experiment_labels[experiment]] = df_results
            df_series_dist_dict[experiment_labels[experiment]] = df_series_dist
            df_series_ws_dict[experiment_labels[experiment]] = df_series_ws

    os.chdir(working_dir)

    """ Consolidate results:
        -   Latency: best, mean and standard deviation
        -   Distance: best, mean and standard deviation
        -   Score: best, mean and standard deviation
        -   Failure rate
    """
    summary = {}
    for key in include:
        label = experiment_labels[key]
        df = df_dict[label]
        latency_data = {
            'best': df.min()['latency'],
            'avg': df.mean()['latency'],
            'std': df.std()['latency']
        }
        df_results = df_results_dict[label]
        dist_data = {
            'best': df_results.min()['closest_distance'],
            'avg': df_results.mean()['closest_distance'],
            'std': df_results.std()['closest_distance']
        }
        ws_data = {
            'best': df_results.min()['weighted_score'],
            'avg': df_results.mean()['weighted_score'],
            'std': df_results.std()['weighted_score']
        }
        speed_data = {
            'best': df_results.min()['closest_approach_speed'],
            'avg': df_results.mean()['closest_approach_speed'],
            'std': df_results.std()['closest_approach_speed']
        }
        pursuer_fuel_data = {
            'best': df_results.min()['closest_approach_pursuer_fuel_usage'],
            'avg': df_results.mean()['closest_approach_pursuer_fuel_usage'],
            'std': df_results.std()['closest_approach_pursuer_fuel_usage']
        }
        fr = df.mean()['failure_2'] * 100

        summary[label] = {
            'failure_rate': fr,
            'latency': latency_data,
            'distance': dist_data,
            'score': ws_data,
            'approach_speed': speed_data,
            'pursuer_fuel_usage': pursuer_fuel_data,
        }

    """ Saves consolidated results in summary file
    """
    summary_filename = "summary.txt"
    with open(summary_filename, "w") as file:
        file.write("RESULTS SUMMARY")
        file.write('\n\n')

        file.write('LATENCY\n')
        file.write('Experiment          \t\tBest\tAvg.\tStd. Dev.\n')
        for key in include:
            label = experiment_labels[key]
            data = summary[label]['latency']
            file.write(f"{label:>25}\t{data['best']:.2f}\t{data['avg']:.2f}\t{data['std']:.2f}\n")
        file.write('\n')

        file.write('SCORE\n')
        file.write('Experiment          \t\tBest\tAvg.\tStd. Dev.\n')
        for key in include:
            label = experiment_labels[key]
            data = summary[label]['score']
            file.write(f"{label:>25}\t{data['best']:.2f}\t{data['avg']:.2f}\t{data['std']:.2f}\n")
        file.write('\n')

        file.write('DISTANCE\n')
        file.write('Experiment          \t\tBest\tAvg.\tStd. Dev.\tFailure rate\n')
        for key in include:
            label = experiment_labels[key]
            data = summary[label]['distance']
            file.write(f"{label:>25}\t{data['best']:.2f}\t{data['avg']:.2f}\t{data['std']:.2f}\t{summary[label]['failure_rate']:.2f} %\n")
        file.write('\n')

        file.write('CLOSEST APPROACH SPEED\n')
        file.write('Experiment          \t\tBest\tAvg.\tStd. Dev.\n')
        for key in include:
            label = experiment_labels[key]
            data = summary[label]['approach_speed']
            file.write(f"{label:>25}\t{data['best']:.2f}\t{data['avg']:.2f}\t{data['std']:.2f}\n")
        file.write('\n')

        file.write('CLOSEST PURSUER FUEL USAGE\n')
        file.write('Experiment          \t\tBest\tAvg.\tStd. Dev.\n')
        for key in include:
            label = experiment_labels[key]
            data = summary[label]['pursuer_fuel_usage']
            file.write(f"{label:>25}\t{data['best']:.2f}\t{data['avg']:.2f}\t{data['std']:.2f}\n")

    font1 = {'family':'serif', 'color': 'red', 'size': 8}
    font2 = {'family':'serif', 'color': 'black', 'size': 8}

    """ Plot latency for each experiment (whisker chat)
    """
    figure = plt.figure(figsize=(8,6))
    bplot_df = pd.DataFrame()
    for key in include:
        key = experiment_labels[key]
        bplot_df = pd.concat([pd.DataFrame({key: df_dict[key]['latency']}), bplot_df], axis=1)
    bplot, bp_dict = bplot_df.boxplot(vert=False, grid=False, showfliers=False, return_type='both')

    # Overlay values
    for line in bp_dict['medians']:
        # get position data for median line
        x, y = line.get_xydata()[1] # top of median line
        # overlay median value
        text(x, y, '%.1f' % x,
             fontdict=font1,
             horizontalalignment='center') # draw above, centered
    for line in bp_dict['boxes']:
        x, y = line.get_xydata()[0] # bottom of left line
        text(x,y, '%.1f' % x,
             fontdict=font2,
             horizontalalignment='right', # right
             verticalalignment='top')      # below
        x, y = line.get_xydata()[3] # bottom of right line
        text(x,y, '%.1f' % x,
             fontdict=font2,
             horizontalalignment='left', # left
             verticalalignment='top')      # below

    bplot.set_xlabel('Latency (ms)')
    figure.tight_layout()
    figure.savefig("fig_latency.png", format='png')
    plt.close(figure)

    """ Plot failure rate for each experiment (bar chat)
    """
    data = []
    for key in include:
        key = experiment_labels[key]
        frequencies = df_dict[key]['failure'].value_counts() / len(df_dict[key]) * 100
        if len(frequencies) > 1:
            data.append([key, frequencies.iloc[1]])
        else:
            data.append([key, 0])
    bplot_df = pd.DataFrame(data, columns=['experiment', 'failure_rate'])

    figure, ax = plt.subplots(figsize=(8, 6))
    plots = sns.barplot(x="failure_rate", y="experiment", orient='h', data=bplot_df)
    for bar in plots.patches:
        plots.annotate("{:.2f}%".format(bar.get_width()),
                       (bar.get_x() + bar.get_width(),
                        bar.get_y() + bar.get_height() / 2), ha='left', va='center',
                       size=8, xytext=(8, 0),
                       textcoords='offset points')
    ax.set_xlim([0, 100])
    plt.xlabel('Average failure rate (%)')
    figure.tight_layout()
    figure.savefig("fig_failure_rate.png", format='png')
    plt.close(figure)

    """ Plot closest distance for each experiment (whisker chat)
    """
    figure = plt.figure(figsize=(8,6))
    bplot_df = pd.DataFrame()
    for key in include:
        key = experiment_labels[key]
        bplot_df = pd.concat([pd.DataFrame({key: df_results_dict[key]['closest_distance']}), bplot_df], axis=1)
    bplot, bp_dict = bplot_df.boxplot(vert=False, grid=False, showfliers=False, return_type='both')

    # Overlay values
    for line in bp_dict['medians']:
        # get position data for median line
        x, y = line.get_xydata()[1] # top of median line
        # overlay median value
        text(x, y, '%.1f' % x,
             fontdict=font1,
             horizontalalignment='center') # draw above, centered
    for line in bp_dict['boxes']:
        x, y = line.get_xydata()[0] # bottom of left line
        text(x,y, '%.1f' % x,
             fontdict=font2,
             horizontalalignment='right', # right
             verticalalignment='top')      # below
        x, y = line.get_xydata()[3] # bottom of right line
        text(x,y, '%.1f' % x,
             fontdict=font2,
             horizontalalignment='left', # left
             verticalalignment='top')      # below

    bplot.set_xlabel('Closest distance (m)')
    figure.tight_layout()
    figure.savefig("fig_closest_distance.png", format='png')
    plt.close(figure)

    """ Plot weighted score for each experiment (whisker chat)
    """
    figure = plt.figure()
    bplot_df = pd.DataFrame()
    for key in include:
        key = experiment_labels[key]
        bplot_df = pd.concat([pd.DataFrame({key: df_results_dict[key]['weighted_score']}), bplot_df], axis=1)
    bplot, bp_dict = bplot_df.boxplot(vert=False, grid=False, showfliers=False, return_type='both')

    # Overlay values
    for line in bp_dict['medians']:
        # get position data for median line
        x, y = line.get_xydata()[1] # top of median line
        # overlay median value
        text(x, y, '%.1f' % x,
             fontdict=font1,
             horizontalalignment='center') # draw above, centered
    for line in bp_dict['boxes']:
        x, y = line.get_xydata()[0] # bottom of left line
        text(x,y, '%.1f' % x,
             fontdict=font2,
             horizontalalignment='right', # right
             verticalalignment='top')      # below
        x, y = line.get_xydata()[3] # bottom of right line
        text(x,y, '%.1f' % x,
             fontdict=font2,
             horizontalalignment='left', # left
             verticalalignment='top')      # below

    bplot.set_xlabel('Score')
    figure.tight_layout()
    figure.savefig("fig_weighted_score.png", format='png')
    plt.close(figure)

    """ Plot distance time series of best run for each experiment (line chat)
    """
    figure = plt.figure()
    for key in df_series_dist_dict:
        if key in best_runs:
            df_csv = df_series_dist_dict[key][best_runs[key]]
            line_chart = plt.plot(df_csv['time'], df_csv['distance'], label=key)
    plt.legend(loc="upper right")
    plt.xlabel('Time (s)')
    plt.ylabel('Distance (m)')
    figure.tight_layout()
    figure.savefig("fig_dist_series.png", format='png')
    plt.close(figure)

    """ Plot weighted score time series of best run for each experiment (line chat)
    """
    figure = plt.figure()
    for key in df_series_ws_dict:
        df_csv = df_series_ws_dict[key][best_runs[key]]
        line_chart = plt.plot(df_csv['time'], df_csv['weighted_score'], label=key)
    plt.legend(loc="upper right")
    plt.xlabel('Time (s)')
    plt.ylabel('Score')
    plt.yscale("log")
    figure.tight_layout()
    figure.savefig("fig_ws_series.png", format='png')
    plt.close(figure)

    if experiment_with_best_run in include:
        """ Plot distance and score time series of best run among all experiments (line chat)
        """
        figure, ax = plt.subplots()
        label = experiment_labels[experiment_with_best_run]
        df_csv = df_series_dist_dict[label][best_runs[label]]
        line_chart = plt.plot(df_csv['time'], df_csv['weighted_score'], label='Score')
        line_chart = plt.plot(df_csv['time'], df_csv['distance'], label='Distance')

        plt.legend(loc="upper right")
        plt.xlabel('Time (s)')
        plt.ylabel('Score / Distance (m)')
        plt.yscale("log")
        plt.title(label)
        figure.tight_layout()
        figure.savefig("fig_dist_ws_series.png", format='png')
        plt.close(figure)


def view_run(run_id):
    """ Args:
        run_id: full path to run id including entity and project
    """
    try:
        api = wandb.Api()
        run = api.run(run_id)
        print("Summary:\n" + str(run.summary))
        print("Config:\n" + str(run.config))
        print("History:\n" + str(run.history()))
        print("Files:")
        for file in run.files():
            print(file.name)

    except Exception as e:
        print("Exception: " + str(e))


if __name__ == '__main__':
    # Load configuration from .env
    dotenv_path = os.path.join(os.path.dirname(__file__), '../../..', 'arclab_mit', 'agents', '.env')
    load_dotenv(dotenv_path)

    wandb_api_key = os.environ['WANDB_API_KEY']
    wandb.login(key=wandb_api_key)

    entity = os.environ['WANDB_ENTITY']
    entity = input(f"entity [{entity}]: ")
    if entity == "":
        entity = os.environ['WANDB_ENTITY']

    project = os.environ['WANDB_PROJECT']
    project = input(f"entity [{project}]: ")
    if project == "":
        project = os.environ['WANDB_PROJECT']

    while True:
        option = input("\nChoose option\ne: export run\ni: import run\nu: upload files\ns: statistics\nv: view run\nq: quit\n")
        option = option.lower()

        if option == 'q':
            break
        if option == "e":
            run_id = input("run id: ")
            run_id = entity + "/" + project + "/" + run_id
            export_run(run_id)
        elif option == "i":
                run_id = input("run id: ")
                experiment = input("run name: ")
                import_run(entity, project, run_id, experiment)
        elif option == "u":
            working_dir = input(f"working directory [{os.getcwd()}]: ")
            if working_dir == "":
                working_dir = os.getcwd()
            path = input("directory or file to upload: ")
            run_id = input("run id: ")
            run_id = entity + "/" + project + "/" + run_id
            upload_files(working_dir, path, run_id)
        elif option == "v":
            run_id = input("run id: ")
            run_id = entity + "/" + project + "/" + run_id
            view_run(run_id)
        elif option == "s":
            working_dir = input(f"working directory [{os.getcwd()}]: ")
            if working_dir == "":
                working_dir = os.getcwd()
            scenario = input("scenario: ")
            statistics = generate_statistics(working_dir, scenario)
        else:
            print("Wrong option: " + option)
