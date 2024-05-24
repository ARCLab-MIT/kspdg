import os
import random


def merge_json_files(input_files, output_file_path):
    with open(output_file_path, 'w') as outfile:
        outfile.write('[\n')
        for i, f in enumerate(input_files):
            with open(f, 'r') as infile:
                lines = infile.readlines()
                # Remove the first and last line which are '[' and ']'
                content_lines = lines[1:-1]
                if i < len(input_files) - 1:
                    outfile.write(''.join(content_lines) + ',\n')  # Add a comma at the end of each file's content
                else:
                    outfile.write(''.join(content_lines) + '\n')  # Do not add a comma for the last file
        outfile.write(']\n')


def split_and_merge_files(input_directory, output_directory, split_ratio=0.50):
    # Create output directories if they don't exist
    os.makedirs(output_directory, exist_ok=True)

    train_val_file_path = os.path.join(output_directory, 'train_val_data.jsonl')
    test_file_path = os.path.join(output_directory, 'test_data.jsonl')

    # Get list of jsonl files
    simple_jsonl_files = [f for f in os.listdir(input_directory) if f.endswith('.jsonl')]
    alpaca_json_files = [os.path.join(input_directory, f) for f in os.listdir(input_directory) if
                         f.endswith('.json') and 'alpaca' in f]
    sharegpt_json_files = [os.path.join(input_directory, f) for f in os.listdir(input_directory) if
                           f.endswith('.json') and 'sharegpt' in f]

    print(len(simple_jsonl_files))

    # Shuffle the files for random splitting
    random.shuffle(simple_jsonl_files)

    # Calculate the number of files for training+validation and test
    num_files = len(simple_jsonl_files)
    num_train_val = int(num_files * split_ratio)
    num_test = num_files - num_train_val

    # Split the files
    train_val_files = simple_jsonl_files[:num_train_val]
    test_files = simple_jsonl_files[num_train_val:]

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

    # Merge alpaca and sharegpt files
    alpaca_output_file_path = os.path.join(output_directory, 'merged_alpaca.json')
    sharegpt_output_file_path = os.path.join(output_directory, 'merged_sharegpt.json')

    merge_json_files(alpaca_json_files, alpaca_output_file_path)
    merge_json_files(sharegpt_json_files, sharegpt_output_file_path)

    print(f"Merged alpaca files into {alpaca_output_file_path}")
    print(f"Merged sharegpt files into {sharegpt_output_file_path}")


directory_name = 'training_ready_data_05-24-24-1'

input_directory = f'./{directory_name}'
output_directory = f'./{directory_name}/split_data'

split_and_merge_files(input_directory, output_directory)
