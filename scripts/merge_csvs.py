import sys
import os
import re
import pandas as pd
import numpy as np


def extract_sample_count(filename):
    # Extract the number of samples from the filename using regex
    match = re.search(r'(\d+)_samples', filename)
    if match:
        return int(match.group(1))
    return -1  # Default to -1 if no match found


def extract_nspw(filename):
    """Extract nspw from the filename using regex."""
    match = re.search(r'nspw=(\d+)', filename)
    if match:
        return int(match.group(1))
    return -1


def extract_n_shots(filename):
    """Extract the list of n_shots from the filename using regex."""
    match = re.findall(r'_(\d+)', filename)
    if match:
        return [int(num) for num in match]
    return []


def compile_npy_files(directory):
    compiled_data = []

    # Iterate over .npy files in the directory
    for file_name in os.listdir(directory):
        if file_name.endswith('.npy'):
            file_path = os.path.join(directory, file_name)

            # Load the .npy file
            data = np.load(file_path)

            nspw = extract_nspw(file_name)
            n_shots_list = extract_n_shots(file_name)

            # Check if the number of rows in data matches the length of n_shots_list
            if data.shape[0] != len(n_shots_list):
                raise ValueError(
                    f"Inconsistent data in {file_name}: expected {len(n_shots_list)} rows but got {data.shape[0]}")

            # Iterate over rows in the data and compile the results
            for i, n_shots in enumerate(n_shots_list):
                for run_num, accuracy in enumerate(data[i]):
                    compiled_data.append({
                        'n_shots': n_shots,
                        'accuracy': accuracy,
                        'run_num': run_num,
                        'nspw': nspw
                    })

    return pd.DataFrame(compiled_data)


def process_directory(root_directory, output_path):
    combined_data = []

    # Step 2: Iterate over directories in the root directory
    for model_name in os.listdir(root_directory):
        model_path = os.path.join(root_directory, model_name)

        # Ensure it's a directory
        if not os.path.isdir(model_path):
            continue

        # Step 3: Iterate over the sub directories in each model's directory
        for dataset_name in os.listdir(model_path):
            dataset_path = os.path.join(model_path, dataset_name)

            # Ensure it's a directory
            if not os.path.isdir(dataset_path):
                continue

            csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]

            # If no CSV files, continue to the next iteration
            if len(csv_files) == 0:
                npy_files = [f for f in os.listdir(dataset_path) if f.endswith('.npy')]
                if len(npy_files) == 0:
                    continue # skip
                df = compile_npy_files(dataset_path)
                df['n_samples'] = -1
            else:

                # If multiple CSV files, select the one with the largest number of samples
                if len(csv_files) > 1:
                    csv_files.sort(key=extract_sample_count, reverse=True)

                csv_path = os.path.join(dataset_path, csv_files[0])

                # Step 5: Read the CSV, add "model", "dataset", and "n_samples" columns
                df = pd.read_csv(csv_path)
                df['n_samples'] = extract_sample_count(csv_files[0])

            df['model'] = model_name
            df['dataset'] = dataset_name

            combined_data.append(df)

    # Step 6: Combine all data and save to output path
    final_df = pd.concat(combined_data, ignore_index=True)
    final_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    root_directory = sys.argv[1]
    output_path = sys.argv[2]
    process_directory(root_directory, output_path)
