#!/bin/bash
#SBATCH --output=/home/joberant/data_nobck/maorivgi/outputs/pcw/%j.out        # redirect stdout
#SBATCH --error=/home/joberant/data_nobck/maorivgi/outputs/pcw/%j.out         # redirect stderr
#SBATCH --partition=killable    # (see next section)
#SBATCH --time=0-23:59:00                     # max time (minutes)
#SBATCH --nodes=1                       # number of machines
#SBATCH --ntasks=1                      # number of processes
#SBATCH --mem=50000
#SBATCH --cpus-per-task=4     # CPU cores per process
#SBATCH --gpus=1               # GPUs in total
#SBATCH --exclude=n-305
#SBATCH --constraint="tesla_v100|geforce_rtx_3090|a5000|a6000"

source ~/.bashrc
conda activate pcw
cd /specific/netapp5/joberant/home/maorivgi/code/Parallel-Context-Windows

# Set environment variables
export ALLENNLP_CACHE_ROOT="/home/joberant/data_nobck/maorivgi/cache"
export CACHE="/home/joberant/data_nobck/maorivgi/cache"
export CODE="/specific/netapp5/joberant/home/maorivgi/code"
export DATA="/specific/netapp5/joberant/home/maorivgi/data"
export NLTK_DATA="/home/joberant/data_nobck/maorivgi/cache/nltk"
export TORCH_HOME="/home/joberant/data_nobck/maorivgi/cache"
export XDG_CACHE_HOME="/home/joberant/data_nobck/maorivgi/cache"

PYTHONPATH=.
OUTPUT_DIR=/home/joberant/data_nobck/maorivgi/outputs/pcw/llama27b/sst2

python run_evaluation.py --dataset sst2 --cache-dir $CACHE --model "meta-llama/Llama-2-7b-hf" --n-windows 1 --n-windows 3 --subsample-test-set 100 --n-runs 3 --output-dir $OUTPUT_DIR --token $HF_TOKEN --n-shots-per-window 1 --n-shots-per-window 3 --n-shots-per-window -1
