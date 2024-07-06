import os
import openai
import time
from dotenv import load_dotenv
from os.path import join, dirname
import wandb
import pandas as pd
import json
import re
import csv

import numpy as np
import ast
from sklearn.metrics import log_loss

from arclab_mit.agents.agent_common import Action
from arclab_mit.agents.fine_tuning_agent import LLMAgent
from arclab_mit.agents.llama_fine_tuning_agent import LlamaAgent


def calculate_loss_accuracy(throttles, predictions):
    # Compile throttles and predictions into one-dimensional numpy arrays
    total_samples = len(throttles)
    throttles_array = np.array([])
    predictions_array = np.array([])
    for i in range(total_samples):
        throttles_array = np.concatenate((throttles_array, np.array(ast.literal_eval(throttles[i]))))
        predictions_array = np.concatenate((predictions_array, np.array(ast.literal_eval(predictions[i]))))

    # Ensure throttles and predictions use integer labels
    throttles_array = throttles_array.astype(int)
    predictions_array = predictions_array.astype(int)

    # Assign prediction probabilities: 1 for predicted label and 0 elsewhere
    total_samples = len(throttles_array)
    predictions_prob = np.zeros((total_samples, 3))
    for i in range(total_samples):
        # Labels are -1, 0 and 1
        predictions_prob[i][predictions_array[i] + 1] = 1

    # Calculate categorical cross-entropy loss
    categorical_cross_entropy_loss = log_loss(throttles_array, predictions_prob, labels=[-1, 0, 1])

    # Calculate accuracy
    correct_predictions = (throttles_array == predictions_array).sum().item()
    accuracy = correct_predictions / total_samples

    return categorical_cross_entropy_loss, accuracy


def evaluate_predictions(datafile):
    print("Evaluate predictions from: " + datafile)
    # Load data file into pandas dataframe
    csv_filename = datafile
    df = pd.read_csv(csv_filename)

    # Open output file
    out_filename = datafile.replace(".csv", ".out")
    out_file = open(out_filename, 'w', newline='')

    throttles = df['throttles']
    predictions = df['prediction']
    total_samples = len(df['throttles'])

    loss, accuracy = calculate_loss_accuracy(throttles, predictions)
    out_file.write("Model predictions:\n")
    out_file.write(f"Categorical cross-entropy loss: {loss:.4f}\n")
    out_file.write(f"Accuracy: {accuracy:.4f}\n")

    naive_predictions = pd.Series(np.full(total_samples, "[1,0,0]"))
    loss, accuracy = calculate_loss_accuracy(throttles, naive_predictions)
    out_file.write("\nNaive predictions:\n")
    out_file.write(f"Categorical cross-entropy loss: {loss:.4f}\n")
    out_file.write(f"Accuracy: {accuracy:.4f}\n")

    out_file.close()


def simulate(datafile):
    print("Simulate run from: " + datafile)
    # Load data file into pandas dataframe
    csv_filename = datafile
    df = pd.read_csv(csv_filename)

    agent = LlamaAgent()

    for _, row in df.iterrows():
        input_data = {k: v for k, v in row.items() if k != 'throttles' and k != 'next_throttles'}

        observation = [input_data["time"], input_data["vehicle_mass"], input_data["vehicle_propellant"],
                       input_data['pursuer_pos_x'], input_data['pursuer_pos_y'], input_data['pursuer_pos_z'],
                       input_data['pursuer_vel_x'], input_data['pursuer_vel_y'], input_data['pursuer_vel_z'],
                       input_data['evader_pos_x'], input_data['evader_pos_y'], input_data['evader_pos_z'],
                       input_data['evader_vel_x'], input_data['evader_vel_y'], input_data['evader_vel_z']]
        sun_position = [input_data['sun_pos_x'], input_data['sun_pos_y'],
                        input_data['sun_pos_z']] if "sun_pos_x" in input_data else None

        vessel_up = np.array([input_data['vessel_up_x'], input_data['vessel_up_y'],
                              input_data['vessel_up_z']]) if "vessel_up_x" in input_data else None

        action = agent.get_action(observation, sun_position, vessel_up)


def generate_predictions_from_jsonl(scenario, model, jsonl_file):
    print("Generating predictions from: " + jsonl_file)
    if scenario.lower().startswith('pe'):
        pattern = os.environ['PE_USER_PROMPT']
    elif scenario.lower().startswith('lbg'):
        pattern = os.environ['LBG_USER_PROMPT']
    else:
        pattern = os.environ['SB_USER_PROMPT']
    pattern = pattern.replace("{CoT}", "")
    pattern = pattern[pattern.find("{"):]
    pattern = pattern.replace("{obs}", "\{(.+?)\}")
    if "{calculations}" in pattern:
        pattern = pattern.replace("{calculations}", "(.*?)")
    add_reasoning = False
    if " Reason step-by-step." in pattern:
        add_reasoning = True
        pattern = pattern.replace(" Reason step-by-step.", "")
    agent = LLMAgent()

    # Initiate the output dataframe
    out_df = pd.DataFrame()

    # Process jsonl file
    with open(jsonl_file, 'r') as file:
        for line in file:
            if line == "" or line == "\n" or line == "\r\n" or line == "\r" or line == "\t" or line == " ":
                # Skip empty lines
                continue
            message_structure = json.loads(line)

            # Pop assistant message and initiate output_data with "throttles" (e.g. action in assistant)
            messages = message_structure['messages']
            assistant = messages.pop()
            action_json = json.loads(assistant['function_call']['arguments'])
            output_data = {
                'throttles': Action.from_enum([action_json['ft'], action_json['rt'], action_json['dt'], action_json['dt'], 0.5])[0:3]
            }
            try:
                # Find last occurrence of "{"
                str = messages[-1]['content']
                p = str.rfind("{")
                k = str[p:]

                # Search for obs and CoT in the last message
                match = re.search(pattern, k)
                if match:
                    # Extract values from the matched groups
                    obs = json.loads('{' + match.group(1) + '}')

                    # Extend output_data with observations
                    output_data.update(obs)

                    messages[-1]['content'] = os.environ['PE_CHAIN_OF_THOUGHT'] + messages[-1]['content']
                    if add_reasoning:
                        messages[-1]['content'] += " Reason step-by-step."
                    # Use LLM model to predict action
                    try:
                        action = agent.check_response(response=agent.get_completion(prompt=messages, model=model))
                    except Exception as e:
                        print("Exception: " + str(e))
                        action = [0, 0, 0, 0.5]

                    # Add prediction to output_data
                    output_data['prediction'] = action[0:3]
                    out_df = pd.concat([out_df, pd.DataFrame([output_data])], ignore_index=True)
            except Exception as e:
                print("Exception: " + str(e))

    basename = os.path.basename(jsonl_file)
    dirname = os.path.dirname(jsonl_file)
    out_filename = join(dirname, "pred_" + basename).replace(".jsonl", ".csv")
    # Writing to CSV file
    with open(out_filename, 'w', newline='') as csvfile:
        # Create a CSV writer object
        csv_writer = csv.writer(csvfile)

        # Write the header (column names) to the CSV file
        csv_writer.writerow(out_df.columns)

        # Write the data rows to the CSV file
        for index, row in out_df.iterrows():
            csv_writer.writerow(row)

    return


def create_fine_tune_job(base_model, training_filename, validation_filename: None,
                         n_epochs: "auto", batch_size: "auto", learning_rate_multiplier: "auto"):
    try:
        # Upload training file
        train_full_response_file = openai.File.create(
            file=open(training_filename,'rb'),
            purpose='fine-tune',
            user_provided_filename=training_filename,
        )
        print(f'training file id: {train_full_response_file.id}')

        # Upload validation file
        validation_file_id = None
        if validation_filename is not None:
            validation_full_response_file = openai.File.create(
                file=open(validation_filename,'rb'),
                purpose='fine-tune',
                user_provided_filename=validation_filename,
            )
            validation_file_id = validation_full_response_file.id
            print(f'validation file id: {validation_file_id}')

        # Create a fine-tuning job
        response = openai.FineTuningJob \
            .create(training_file=train_full_response_file.id,
                    validation_file=validation_file_id,
                    model=base_model,
                    suffix='KSPGPT',
                    hyperparameters={'n_epochs': n_epochs,
                                     'batch_size': batch_size,
                                     'learning_rate_multiplier': learning_rate_multiplier})
        job_id = response.id

        print("Created fine tunning job: " + job_id)
        print(response)

        with open('results/result_' + job_id + '.txt', 'w') as result:
            result.write ("Created fine tunning job:")
            result.write (str(response))
            status_completion_list = ['succeeded', 'failed', 'cancelled']
            while True:
                job = openai.FineTuningJob.retrieve(job_id)
                if job.status in status_completion_list:
                    print("\nJob terminated")
                    print(job)
                    result.write("\n\nJob result:")
                    result.write(str(job))
                    break
                time.sleep(10)
    except Exception as e:
        print("Exception: " + str(e))


def retrieve_fine_tune_job(job_id):
    try:
        job = openai.FineTuningJob.retrieve(job_id)
        print(job)
    except Exception as e:
        print ("Exception: " + str(e))


def log_job_results(scenario, job_id, experiment, generate_predictions):
    try:
        # Read result file
        response = openai.FineTuningJob.retrieve(job_id)
        if response['status'].lower() != 'succeeded':
            print("Job ended with status " + response['status'])
            return
        training_file = response['training_file']
        validation_file = response['validation_file']
        result_files = response['result_files']
        n_epochs = response['hyperparameters']['n_epochs']
        batch_size = response['hyperparameters']['batch_size']
        learning_rate_multiplier = response['hyperparameters']['learning_rate_multiplier']

        r = openai.File.retrieve(result_files[0])
        content = openai.File.download(result_files[0]).decode()

        lines = content.split('\n')
        columns = lines[0].split(',')
        df = pd.DataFrame([row.split(',') for row in lines[1:]],
                          columns=columns)
        n_steps = len(df)-1
        n_steps_per_epoch = int(n_steps / n_epochs)

        entity = os.environ["WANDB_ENTITY"]
        project = os.environ["WANDB_PROJECT"]
        wandb.init(
            # Set the entity where this run will be logged
            entity=entity,
            # Set the project where this run will be logged
            project=project,
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=experiment,
            job_type = "LLM_fine_tune",
            # Track hyperparameters and run metadata
            config={
                "architecture": "LLM",
                "job_id": job_id,
                "model": response['model'],
                "training_file": response['training_file'],
                "validation_file": response['validation_file'],
                "fine-tuned model": response['fine_tuned_model'],
                "epochs": n_epochs,
                "batch_size": batch_size,
                "learning_rate_multiplier": learning_rate_multiplier,
            })
        print("Logging metrics ...")
        i = 0
        for epoch in range(n_epochs):
            for step in range(n_steps_per_epoch):
                metrics = {'epoch': epoch,
                           'step': i,
                           'training_loss': float(df['train_loss'][i]),
                           'training_token_accuracy': float(df['train_accuracy'][i])}
                if df['valid_loss'][i] is not None and df['valid_loss'][i] != '':
                    val_metrics = {
                        'validation_loss': float(df['valid_loss'][i]),
                        'validation_token_accuracy': float(df['valid_mean_token_accuracy'][i])}
                    metrics.update(val_metrics)
                wandb.log({**metrics})
                i += 1
        print("... done")

        """ Create dataset subdirectory if it does not exist """
        job_dir = job_id
        dataset_dir = join(job_id, 'dataset')
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        """ Save job results"""
        with open(join(job_dir, 'result.txt'), 'w') as file:
            file.write(f"Experiment: {experiment}\n")
            file.write(f"Job result:\n")
            file.write(str(response))

        """" Download training and validation files """
        training = openai.File.download(training_file).decode()
        validation = openai.File.download(validation_file).decode()
        result_files = openai.File.download(result_files[0]).decode()
        with open(join(dataset_dir, 'training.jsonl'), 'w') as file:
            for line in training.split('\n'):
                file.write(line)
#                file.write(line + '\n')
        print("Downloaded training file: " + join(dataset_dir, 'training.jsonl'))
        with open(join(dataset_dir, 'validation.jsonl'), 'w') as file:
            for line in validation.split('\n'):
                file.write(line)
#                file.write(line + '\n')
        print("Downloaded validation file: " + join(dataset_dir, 'validation.jsonl'))
        with open(join(dataset_dir, 'result_files.csv'), 'w') as file:
            for line in result_files.split('\n'):
                file.write(line)
#                file.write(line + '\n')
        print("Downloaded result file: " + join(dataset_dir, 'result_files.csv'))

        if generate_predictions:
            """ Generate predictions """
            model = response['fine_tuned_model']
            generate_predictions_from_jsonl(scenario, model, join(dataset_dir, 'training.jsonl'))
            generate_predictions_from_jsonl(scenario, model, join(dataset_dir, 'validation.jsonl'))

            """ Evaluate predictions """
            evaluate_predictions(join(dataset_dir, 'pred_training.csv'))
            evaluate_predictions(join(dataset_dir, 'pred_validation.csv'))

        """ Get run id and retrieve run """
        api = wandb.Api()
        run_id = entity + '/' + project + '/' + wandb.run.id
        run = api.run(run_id)

        """  Add training, validation and prediction files to wandb run
        """
        filename = join(job_dir, 'result.txt')
        run.upload_file(filename)
        print("Saved: " + filename)
        for filename in os.listdir(dataset_dir):
            filename = join(dataset_dir, filename)
            run.upload_file(filename)
            print("Saved: " + filename)

        # Mark the run as finished
        wandb.finish()

    except Exception as e:
        print("Exception: " + str(e))


if __name__ == '__main__':
    # Load configuration from .env
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'agents', '.env')
    load_dotenv(dotenv_path)

    # Load prompts from alex_prompts.v2
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'agents', 'alex_prompts.v2')
    load_dotenv(dotenv_path)

    openai.api_key = os.environ['OPENAI_API_KEY']
    wandb_api_key = os.environ['WANDB_API_KEY']
    wandb.login(key=wandb_api_key)

    while True:
        option = input("\nChoose option:\nc: create fine-tune job\nr: retrieve fine-tune job\nl: log fine-tune job\np: predict\ns: simulate\nq: quit\n")
        option = option.lower()

        try:
            if option == 'q':
                break
            elif option == 'c':
                model = "gpt-3.5-turbo-0125"
                if os.environ['BASE_MODEL'] != '':
                    model = os.environ['BASE_MODEL']
                data = input(f"base model [{model}]: ")
                if data != '':
                    model = data
                train = input("train file: ")
                validation = None
                data = input("validation file: ")
                if data != '':
                    validation = data
                n_epochs = "auto"
                data = input(f"# epochs [{n_epochs}]: ")
                if data != '':
                    n_epochs = data
                batch_size = "auto"
                data = input(f"batch size [{batch_size}]: ")
                if data != '':
                    batch_size = data
                learning_rate_multiplier = "auto"
                data = input(f"learning rate multiplier [{learning_rate_multiplier}]: ")
                if data != '':
                    learning_rate_multiplier = data

                print("\n")
                print("base model: " + model)
                print("train file: " + train)
                if validation is not None:
                    print("validation file: " + validation)
                print("# epochs: " + n_epochs)
                print("batch size: " + batch_size)
                print("learning_rate_multiplier: " + learning_rate_multiplier)
                data = input("\nIs this information correct [y/n]: ")
                if data.lower() == 'y':
                    create_fine_tune_job(model, train, validation, n_epochs, batch_size, learning_rate_multiplier)
            elif option == "r":
                job_id = input("job id: ")
                retrieve_fine_tune_job(job_id)
            elif option == "l":
                scenario = input("scenario: ")
                job_id = input("job id: ")
                experiment = input("experiment: ")
                tmp = input("generate predictions [y/n]: ")
                generate_predictions = (tmp.lower() == "y")
                log_job_results(scenario, job_id, experiment, generate_predictions )
            elif option == "e":
                datafile = input("datafile (csv): ")
                evaluate_predictions(datafile)
            elif option == 'p':
                model = os.environ['MODEL']
                scenario = os.environ['SCENARIO']
                data = input(f"scenario [{scenario}]: ")
                if data != '':
                    scenario = data
                jsonl_file = input("jsonl file: ")
                generate_predictions_from_jsonl(scenario, model, jsonl_file)
                evaluate_predictions(jsonl_file.replace(".jsonl", ".csv"))
            elif option == 's':
                datafile = input("datafile (csv): ")
                simulate(datafile)
            else:
                print("Wrong option: " + option)
        except Exception as e:
            print("Exception: " + str(e))
