import argparse
import logging
from typing import List, Optional

import pandas as pd
from transformers import PreTrainedTokenizerBase
import numpy as np

from datasets_loader import DATASET_NAMES2LOADERS
from experiment_manager import ExperimentManager
from model_loaders import load_pcw_wrapper
from utils import get_max_n_shots, filter_extremely_long_samples, save_results
import os

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')


def get_dataset(dataset: str, tokenizer: PreTrainedTokenizerBase) -> (pd.DataFrame, pd.DataFrame, List):
    da = DATASET_NAMES2LOADERS[dataset]()
    # Filter extremely long samples from both train and test samples:
    _logger.info("filtering test set:")
    test_df = filter_extremely_long_samples(da.test_df, tokenizer)
    _logger.info("filtering train set:")
    train_df = filter_extremely_long_samples(da.train_df, tokenizer)
    return test_df, train_df, da.labels


def run_pcw_experiment(datasets: List[str], models: List[str], cache_dir: str, subsample_test_set: int, output_dir: str,
                       n_windows: List[int], n_shots_per_window: List[int], n_runs: int,
                       random_seed: int, right_indentation: bool, token=None, overwrite=False) -> None:
    base_output_dir = output_dir
    all_records = []
    for model in models:
        clean_model_name = model.replace('/', '+').replace(' ', '_')
        print(f'* Starting with model: {model} ({clean_model_name})')
        for dataset in datasets: 
            clean_dataset_name = dataset.replace('/', '+').replace(' ', '_')
            print(f'\t- Running with dataset: {dataset} ({clean_dataset_name})')
            output_dir = os.path.join(base_output_dir, clean_model_name, clean_dataset_name)
            pcw_model = load_pcw_wrapper(model, cache_dir, right_indentation, max(n_windows), token=token)

            test_df, train_df, labels = get_dataset(dataset, pcw_model.tokenizer)

            records = []

            for nspw in n_shots_per_window:
                n_shots = [i * nspw for i in n_windows]  # here nspw may still be -1
                output_path = os.path.join(output_dir, f"nspw={nspw}-n_shots_results_{'_'.join([str(i) for i in n_shots])}.npy")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                if os.path.exists(output_path) and not overwrite:
                    print(f'Found results in {output_path}. Skipping...')
                    
                    # just load the records before
                    # load the npy file
                    accuracies = np.load(output_path)
                    with open(os.path.join(output_dir, f"nspw={nspw}-n_shots.txt"), 'r') as f:
                        n_shots = eval(f.read())
                    rows, cols = accuracies.shape
                    for i in range(rows):
                        for j in range(cols):
                            record = {
                                "n_shots": n_shots[i],
                                "accuracy": accuracies[i][j],
                                "run_num": j,
                                "nspw": nspw
                            }
                            records.append(record)

                    continue
                elif os.path.exists(output_path) and overwrite:
                    print(f'Found results in {output_path}. Overwriting...')
                else:
                    print(f'Running with {output_path}...')

                if nspw == -1:
                    # default behavior: we take the maximum number of samples per window
                    nspw = get_max_n_shots(train_df, test_df, pcw_model.tokenizer, pcw_model.context_window_size)
                    _logger.info(f"Found max n shot per window = {nspw}")
                print(f'Running with NSPW={nspw}, and n_windows={n_windows}')

                n_shots = [i * nspw for i in n_windows]
                with open(os.path.join(output_dir, f"nspw={nspw}-n_shots.txt"), 'w') as f:
                    f.write(str(n_shots))

                em = ExperimentManager(test_df, train_df, pcw_model, labels, random_seed=random_seed,
                                    n_shots_per_window=nspw, subsample_test_set=subsample_test_set)

                accuracies = em.run_experiment_across_shots(n_shots, n_runs)  # an ndarry of shape (n_runs, len(n_shots))
                save_results(dataset, n_shots, accuracies, output_path, model, plot_results=False)

                rows, cols = accuracies.shape

                for i in range(rows):
                    for j in range(cols):
                        record = {
                            "n_shots": n_shots[i],
                            "accuracy": accuracies[i][j],
                            "run_num": j,
                            "nspw": nspw
                        }
                        records.append(record)

            # assume output dir already contains the model name
            fname = f"{output_dir}/n_shots_results_over_{subsample_test_set}_samples_seed_{random_seed}_ri={right_indentation}.csv"
            pd.DataFrame(records).to_csv(fname, index=False)
            print('---------------------------------------------------')
            print(f'Done running model {model} on dataset {dataset}. You can find the results in {fname}')
            
            all_records.extend([r | {'model': model, 'dataset': dataset} for r in records])  # require python 3.9+
    fname = f"{base_output_dir}/all_results_over_{subsample_test_set}_samples_seed_{random_seed}_ri={right_indentation}.csv"
    pd.DataFrame(all_records).to_csv(fname, index=False)
    print('---------------------------------------------------')
    print(f'Done running all models on all datasets. You can find the results in {fname}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='datasets', action='append', required=True,
                        help=f'Name of dataset (for example sst2).'
                             f' The supported datasets are: {DATASET_NAMES2LOADERS.keys()}')
    parser.add_argument('--model', dest='models', action='append', default='gpt2',
                        help='HF model name to use, either gpt2 or LLaMa family models')
    parser.add_argument('--subsample-test-set', dest='subsample_test_set', action='store', required=False, type=int,
                        help='Size of test set to use to speed up eval. None means using all test set.')
    parser.add_argument('--output-dir', dest='output_dir', required=False, help="Directory for saving the results",
                        default='./temp', action='store', type=str)
    parser.add_argument('--cache-dir', help="Hugging face cache dir", type=str, default=None, dest='cache_dir')
    parser.add_argument('--random-seed', dest='random_seed', required=False, default=42, action='store', type=int)
    parser.add_argument('--n-runs', dest='n_runs',
                        help="Number of times experiments are repeated for every number of windows", action='store',
                        type=int, default=1)
    parser.add_argument('-n', '--n-windows', dest='n_windows', help="Number of parallel context windows",
                        action='append', type=int)
    parser.add_argument('--n-shots-per-window', dest='n_shots_per_window',
                        help="number of examples to fit in each window (can be multiple items). use -1 for maximum possible",
                        type=int, required=True, action='append')
    parser.add_argument('--right-indentation', dest='right_indentation', help="ident all windows to the right",
                        action='store_true', default=False)
    parser.add_argument('--overwrite', dest='overwrite', help="If true, overwrite existing results",
                        action='store_true', default=False)
    parser.add_argument('--token', dest='token', default=None, type=str, help='HF token if needed')
    args = parser.parse_args()
    print('running with token:', args.token)
    run_pcw_experiment(**vars(args))
