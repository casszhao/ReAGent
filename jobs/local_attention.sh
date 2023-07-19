#!/bin/bash

module load Anaconda3/2022.10
module load cuDNN/8.0.4.30-CUDA-11.1.1

# Activate env
source activate eva      # via conda
# source .venv/bin/activate           # via venv

# this script is for the 3 attention

model_name="gpt2-medium"
FA_name="rollout_attention" # select from all_attention rollout_attention last_attention

    
importance_results="rationalization_results/analogies/"$model_name"_"$FA_name



# Run rationalization task
mkdir -p $importance_results
mkdir -p $logpath
python src/rationalization/run_analogies.py \
    --rationalization-config config/"$model_name"_"$FA_name".json \
    --model $model_name \
    --tokenizer $model_name \
    --data-dir data/analogies \
    --importance_results_dir $importance_results \
    --device cuda \
    --logfile "logs/analogies/"$model_name"_"$FA_name"_extracting.log" \
    --input_data_size 160


# Evaluate results. This can be done on a local machine
eva_output_dir="evaluation_results/analogies/"$model_name"_"$FA_name
mkdir -p $eva_output_dir
python src/evaluation/evaluate_analogies.py \
    --importance_results_dir $importance_results \
    --eva_output_dir $eva_output_dir \
    --model $model_name \
    --tokenizer $model_name \
    --logfile "logs/analogies/"$model_name"_"$FA_name"_eva.log" 