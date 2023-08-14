#!/bin/bash

cache_dir="cache/"
model_name="gpt2-medium"
model_short_name="gpt2"
hyper=""

FA_name="inseq_gradient_shap"

experiment_id=$FA_name$hyper

logfolder_shortname=logs/analogies/$model_short_name"_"$experiment_id
mkdir -p $logfolder_shortname
importance_results="rationalization_results/analogies/"$model_short_name"_"$experiment_id
mkdir -p $importance_results

python src/rationalization/run_analogies.py \
    --rationalization-config config/$experiment_id.json \
    --model $model_name \
    --tokenizer $model_name \
    --data-dir data/analogies/$model_short_name/ \
    --importance_results_dir $importance_results \
    --device cuda \
    --logfolder $logfolder_shortname \
    --input_num_ratio 1 \
    --cache_dir $cache_dir
