#!/bin/bash
#SBATCH --comment=seq_rationales_test-1
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=jobs.out/%j.log
#SBATCH --time=4-00:00:00
#SBATCH --mail-user=zhixue.zhao@sheffield.ac.uk

# Load modules & activate env

module load Anaconda3/2022.10
module load cuDNN/8.0.4.30-CUDA-11.1.1

# Activate env
source activate seq      # via conda
# source .venv/bin/activate           # via venv

# Generate evaluation data set (Only need to be done once)
# mkdir -p data/analogies
# python src/data/prepare_evaluation_analogy.py \
#     --analogies-file data/analogies.txt \
#     --output-dir data/analogies \
#     --compact-output True \
#     --schema-uri ../../docs/analogy.schema.json \
#     --device cuda
    

# Run rationalization task
mkdir -p rationalization_results/analogies/gpt2_medium_attention
mkdir -p logs/analogies/gpt2_medium_attention
python src/rationalization/random_replacing/run_analogies.py \
    --rationalization-config config/cass.json \
    --model gpt2-medium \
    --tokenizer gpt2-medium \
    --data-dir data/analogies \
    --output-dir rationalization_results/analogies/gpt2-medium.sampling.uniform \
    --device cuda \
    --logfile logs/analogies/gpt2-medium.sampling.uniform/test.log \
    --input_data_size 1 \

# Migrate baseline results (Only need to be done once for each approach)
# mkdir -p rationalization_results/analogies/gpt2-medium.last_attention
# python src/rationalization/random_replacing/migrate_results_analogies.py \
#     --data-dir data/analogies \
#     --input-dir rationalization_results/analogies-old/last_attention \
#     --output-dir rationalization_results/analogies/gpt2-medium.last_attention \
#     --tokenizer gpt2-medium 

# # Evaluate results. This can be done on a local machine
# mkdir -p evaluation_results/analogies/
# python src/rationalization/random_replacing/evaluate_analogies.py \
#     --data-dir data/analogies \
#     --target-dir rationalization_results/analogies/gpt2-medium.sampling.uniform \
#     --baseline-dir rationalization_results/analogies/gpt2-medium.last_attention \
#     --output-path evaluation_results/analogies/gpt2-medium.sampling.uniform.csv \
#     --tokenizer gpt2-medium 

