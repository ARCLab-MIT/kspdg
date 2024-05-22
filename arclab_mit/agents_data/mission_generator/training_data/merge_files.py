import os
import random


def split_and_merge_files(input_directory, output_directory, split_ratio=0.50):
    # Create output directories if they don't exist
    os.makedirs(output_directory, exist_ok=True)

    train_val_file_path = os.path.join(output_directory, 'train_val_data.jsonl')
    test_file_path = os.path.join(output_directory, 'test_data.jsonl')

    # Get list of jsonl files
    jsonl_files = [f for f in os.listdir(input_directory) if f.endswith('.jsonl')]

    print(len(jsonl_files))

    # Shuffle the files for random splitting
    random.shuffle(jsonl_files)

    # Calculate the number of files for training+validation and test
    num_files = len(jsonl_files)
    num_train_val = int(num_files * split_ratio)
    num_test = num_files - num_train_val

    # Split the files
    train_val_files = jsonl_files[:num_train_val]
    test_files = jsonl_files[num_train_val:]

    print(f"Total files: {num_files}")
    print(f"Training+Validation files: {num_train_val}")
    print(f"Test files: {num_test}")

    # Merge training+validation files into one
    with open(train_val_file_path, 'w') as train_val_outfile:
        for f in train_val_files:
            with open(os.path.join(input_directory, f), 'r') as infile:
                for line in infile:
                    train_val_outfile.write(line)

    # Merge test files into one
    with open(test_file_path, 'w') as test_outfile:
        for f in test_files:
            with open(os.path.join(input_directory, f), 'r') as infile:
                for line in infile:
                    test_outfile.write(line)

    print(f"Merged {len(train_val_files)} files into {train_val_file_path}")
    print(f"Merged {len(test_files)} files into {test_file_path}")


input_directory = './training_ready_data_05-22-24-1'
output_directory = './training_ready_data_05-22-24-1/split_data'

split_and_merge_files(input_directory, output_directory)
