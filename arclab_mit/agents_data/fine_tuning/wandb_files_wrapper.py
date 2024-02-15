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
    """ Args:
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


def import_run(emtity, project, run_id, experiment):
    """ Args:
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
    """ Args:
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

def generate_statistics_experiment(working_dir, experiment):
    """ Args:
        working_dir: working directory for path
        experiment: relative poth to local file or directory
    """
    print("Generating statistics for experiment: " + experiment)
    try:
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

        d = {'latency': [], 'failure': []}
        for filename in filenames:
            for line in open(filename, 'r'):
                data = json.loads(line)
                d['latency'].append(data['end_time_ms'] - data['start_time_ms'])
                try:
                    d['failure'].append(data['outputs']['choices'][0]['message']['function_call']['name'] not in ['perform_action'])
                except Exception as e:
                    d['failure'].append(True)
        df = pd.DataFrame(d, columns=['latency', 'failure'])

        df_series_dist = None
        df_series_ws = None
        pe1Env = PursuitEvadeGroup1Env("pe1_i1_init")
        for filename in csv_filenames:
            # Uncomment next line when files use correct headers
            # df_csv = pd.read_csv(filename)
            df_csv = pd.read_csv(filename, skiprows=1, header=None)

            # Set column names
            if len(df_csv.columns) == 16:
                columns = [
                    'throttles',
                    'time',
                    'vehicle_mass',
                    'vehicle_propellant',
                    'pursuer_pos_x',
                    'pursuer_pos_y',
                    'pursuer_pos_z',
                    'pursuer_vel_x',
                    'pursuer_vel_y',
                    'pursuer_vel_z',
                    'evader_pos_x',
                    'evader_pos_y',
                    'evader_pos_z',
                    'evader_vel_x',
                    'evader_vel_y',
                    'evader_vel_z'
                ]
            else:
                columns = [
                    'throttles',
                    'duration',
                    'time',
                    'vehicle_mass',
                    'vehicle_propellant',
                    'pursuer_pos_x',
                    'pursuer_pos_y',
                    'pursuer_pos_z',
                    'pursuer_vel_x',
                    'pursuer_vel_y',
                    'pursuer_vel_z',
                    'evader_pos_x',
                    'evader_pos_y',
                    'evader_pos_z',
                    'evader_vel_x',
                    'evader_vel_y',
                    'evader_vel_z'
                ]

            if len(df_csv.columns) == 18:
                columns += ['weighted_score']
            if len(df_csv.columns) > 18:
                columns += [
                    'guard_pos_x',
                    'guard_pos_y',
                    'guard_pos_z',
                    'guard_vel_x',
                    'guard_vel_y',
                    'guard_vel_z'
                ]
                if len(df_csv.columns) == 24:
                    columns += ['weighted_score']
            df_csv.columns = columns
            if 'weighted_score' in df_csv.columns:
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

            if df_series_ws is None:
                df_series_ws = df_csv

            # Check dataframe contains weighted_score column
            if 'weighted_score' in df_csv.columns:
                figure = plt.figure()
                line_chart = plt.plot(df_csv['time'], df_csv['weighted_score'])
                plt.xlabel('time (s)')
                plt.ylabel('weighted score')
                plt.title(experiment)
                figure.savefig("fig_ws_series_" + os.path.basename(filename) + ".png", format='png')
                plt.close(figure)

            closest_distance = sys.float_info.max
            data = []
            for index, row in df_csv.iterrows():
                distance = np.linalg.norm([row['evader_pos_x']-row['pursuer_pos_x'],
                                           row['evader_pos_y']-row['pursuer_pos_y'],
                                           row['evader_pos_z']-row['pursuer_pos_z']], ord=2)
                if distance < closest_distance:
                    closest_distance = distance
                data.append(closest_distance)
            df_csv['closest_distance'] = data
            if df_series_dist is None:
                df_series_dist = df_csv

            figure = plt.figure()
            line_chart = plt.plot(df_csv['time'], df_csv['closest_distance'])
            plt.xlabel('time (s)')
            plt.ylabel('closest distance (m)')
            plt.title(experiment)
            figure.savefig("fig_dist_series_" + os.path.basename(filename) + ".png", format='png')
            plt.close(figure)

        d_results = {'closest_distance': [], 'weighted_score': []}
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
        df_results = pd.DataFrame(d_results, columns=['closest_distance', 'weighted_score'])

        statistics_filename = "statistics.txt"
        with open(statistics_filename, "w") as file:
            file.write("Run statistics:\n" + df.describe(include='all').to_string())
            file.write("\n\nResults statistics:\n" + df_results.describe(include='all').to_string())

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

        bplot.set_ylabel('latency (ms)')
        figure.tight_layout()
        figure.savefig("fig_latency.png", format='png')
        plt.close(figure)

        figure = plt.figure(figsize=(8,6))
        frequencies = df['failure'].value_counts() / len(df) * 100
        frequencies = round(frequencies, 2)
        bar_chart = plt.bar(np.arange(len(frequencies)), frequencies)
        for rect1 in bar_chart:
            height = rect1.get_height()
            plt.annotate("{}%".format(height), (rect1.get_x() + rect1.get_width() / 2, height + .05), ha="center",
                         va="bottom", fontsize=15)
        plt.xticks([0, 1], ["Success", "Failure"])
        plt.ylabel('frequency')
        figure.tight_layout()
        figure.savefig("fig_failure.png", format='png')
        plt.close(figure)

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

        bplot.set_ylabel('score')
        figure.tight_layout()
        figure.savefig("fig_weighted_score.png", format='png')
        plt.close(figure)

    except Exception as e:
        print ("Exception: " + str(e))
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    return df, df_results, df_series_dist, df_series_ws


def generate_statistics(working_dir, scenario):
    experiments = [ os.path.basename(f.path) for f in os.scandir(working_dir)
                    if f.is_dir() and os.path.basename(f.path).startswith(scenario)]

    experiment_labels = {
        'PE1_LLM': 'Baseline LLM',
        'PE1_navball_LLM': 'Baseline LLM (w/ navball)',
        'PE1_w_o_system_prompt_lrm_2': 'Fine tuning',
        'PE1_w_o_system_prompt_lrm_0.2': '+ hyperparameter tuning',
        'PE1_w_system_prompt_lrm_0.2': '+ system prompt',
        'PE1_w_system_prompt_multfiles_lrm_0.2': '+ two training gameplays',
        'PE1_GamePlay_1': 'Gameplay 1',
        'PE1_GamePlay_2': 'Gameplay 2',
    }
    include = ['PE1_LLM',
               'PE1_w_o_system_prompt_lrm_2',
               'PE1_w_o_system_prompt_lrm_0.2',
               'PE1_w_system_prompt_lrm_0.2',
               'PE1_w_system_prompt_multfiles_lrm_0.2']

    include = ['PE1_LLM',
               'PE1_navball_LLM']

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

    font1 = {'family':'serif', 'color': 'red', 'size': 8}
    font2 = {'family':'serif', 'color': 'black', 'size': 8}

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

    bplot.set_xlabel('latency (ms)')
    figure.tight_layout()
    figure.savefig("fig_latency.png", format='png')
    plt.close(figure)

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

    bplot.set_xlabel('closest distance (m)')
    figure.tight_layout()
    figure.savefig("fig_closest_distance.png", format='png')
    plt.close(figure)

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
    plt.xlabel('average failure rate (%)')
    figure.tight_layout()
    figure.savefig("fig_failure_rate.png", format='png')
    plt.close(figure)

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

    bplot.set_xlabel('score')
    figure.tight_layout()
    figure.savefig("fig_weighted_score.png", format='png')
    plt.close(figure)

    figure = plt.figure()
    for key in df_series_dist_dict:
        df_csv = df_series_dist_dict[key]
        line_chart = plt.plot(df_csv['time'], df_csv['closest_distance'], label=key)
    plt.legend(loc="upper right")
    plt.xlabel('time (s)')
    plt.ylabel('closest distance')
    figure.tight_layout()
    figure.savefig("fig_dist_series.png", format='png')
    plt.close(figure)

    figure = plt.figure()
    for key in df_series_ws_dict:
        df_csv = df_series_ws_dict[key]
        line_chart = plt.plot(df_csv['time'], df_csv['weighted_score'], label=key)
    plt.legend(loc="upper right")
    plt.xlabel('time (s)')
    plt.ylabel('weighted score')
    figure.tight_layout()
    figure.savefig("fig_ws_series.png", format='png')
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
    wandb_api = wandb.Api()

    entity = os.environ['WANDB_ENTITY']
    entity = input(f"entity [{entity}]: ")
    if entity == "":
        entity = os.environ['WANDB_ENTITY']

    project = os.environ['WANDB_PROJECT']
    project = input(f"entity [{project}]: ")
    if project == "":
        project = os.environ['WANDB_PROJECT']

    while True:
        option = input("\nChoose option\ne: export run\ni: import run\nu: upload files\ns: statistics\nv: view run\nd: delete run\nq: quit\n")
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
        elif option == "d":
            run_id = input("run id: ")
            confirmation = input("Delete run_id " + run_id + " [y/n]: ")
            if confirmation.lower() == 'y':
                run = wandb_api.run("carrusk/KSPDG Challenge (paper)/" + run_id)
                run.delete()
        else:
            print("Wrong option: " + option)
