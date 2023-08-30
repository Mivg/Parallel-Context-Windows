#!/bin/bash
#SBATCH --output=/home/joberant/data_nobck/maorivgi/outputs/pcw/%j.out        # redirect stdout
#SBATCH --error=/home/joberant/data_nobck/maorivgi/outputs/pcw/%j.out         # redirect stderr
#SBATCH --partition=gpu-a100-killable   
#SBATCH --time=0-2:00:00                     # max time (minutes)
#SBATCH --nodes=1                       # number of machines
#SBATCH --ntasks=1                      # number of processes
#SBATCH --mem=50000
#SBATCH --cpus-per-task=4     # CPU cores per process
#SBATCH --gpus=1               # GPUs in total
#SBATCH --exclude=n-305
#SBATCH --constraint="a100"

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
OUTPUT_DIR=/home/joberant/data_nobck/maorivgi/outputs/pcw/

python run_evaluation.py \
--datasets sst2 SetFit/sst5 banking77 ibm/clinic150-sur trec \
--model gpt2-large gpt2-xl "meta-llama/Llama-2-7b-hf" "meta-llama/Llama-2-13b-hf" \
--n-windows 1 2 3 4 5 6 7 8 9 10 15 \
--subsample-test-set 250 \
--n-runs 10 \
--output-dir $OUTPUT_DIR \
--n-shots-per-window 1 2 3 4 5 10 15 20 -1 \
--cache-dir $CACHE \
--token $HF_TOKEN