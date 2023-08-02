#!/bin/bash

cache_dir="cache/"
model_name="KoboldAI/OPT-6.7B-Erebus"
model_short_name="OPT6B"
hyper=""

FA_name="inseq_gradient_shap"

experiment_id=$FA_name$hyper

logfolder_shortname=logs/analogies/$model_short_name"_"$experiment_id
mkdir -p $logfolder_shortname
importance_results="rationalization_results/analogies/"$model_short_name"_"$experiment_id
mkdir -p $importance_results
eva_output_dir="evaluation_results/analogies/"$model_short_name"_"$experiment_id
mkdir -p $eva_output_dir

# ratio

for rationale_ratio_for_eva in 0.05 0.1 0.2 0.3 1
do
echo "  for rationale "
echo $rationale_ratio_for_eva
python src/evaluation/evaluate_analogies.py \
    --importance_results_dir $importance_results \
    --eva_output_dir $eva_output_dir \
    --model $model_name \
    --tokenizer $model_name \
    --logfolder $logfolder_shortname \
    --rationale_size_ratio $rationale_ratio_for_eva \
    --cache_dir $cache_dir
done

# flex

python src/evaluation/evaluate_analogies.py \
    --importance_results_dir $importance_results \
    --eva_output_dir $eva_output_dir \
    --model $model_name \
    --tokenizer $model_name \
    --logfolder $logfolder_shortname \
    --rationale_size_ratio 0 \
    --rational_size_file "rationalization_results/analogies-greedy-lengths.json" \
    --cache_dir $cache_dir
