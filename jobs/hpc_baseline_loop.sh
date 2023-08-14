#!/bin/bash
#SBATCH --comment=gpt2
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=jobs.out/%j.log
#SBATCH --time=4-00:00:00
#SBATCH --mail-user=zhixue.zhao@sheffield.ac.uk

#$ -N gpt2
#SBATCH --job-name=optBASE


# Load modules & activate env

module load Anaconda3/2022.10
module load CUDA/11.8.0

# Activate env
source activate dev-inseq      # via conda

cache_dir="cache/"

model_name="gpt2-xl"
# "gpt2-medium"
# "gpt2-xl"
# "EleutherAI/gpt-j-6b"
# "facebook/opt-350m"
# "facebook/opt-1.3b"
# "KoboldAI/OPT-6.7B-Erebus"
model_short_name="gpt2_xl" 
# gpt2 gpt2_xl gpt6b
# OPT350M OPT1B OPT6B




##########  selecting FA

# select from: "attention_rollout" "attention_last" "attention"
# select from: "norm" "gradient_shap" "integrated_gradients" "input_x_gradient" 
FA_name="attention_rollout" 

importance_results="rationalization_results/analogies/"$model_short_name"_"$FA_name
eva_output_dir="evaluation_results/analogies/"$model_short_name"_"$FA_name
mkdir -p $importance_results
mkdir -p $eva_output_dir
mkdir -p logs/analogies/$model_name"_"$FA_name
logfolder=logs/analogies/$model_name"_"$FA_name
mkdir -p logs/analogies/$model_short_name"_"$FA_name
logfolder_shortname=logs/analogies/$model_short_name"_"$FA_name


# # Generate evaluation data set (Only need to be done once)
# mkdir -p "data/analogies/"$model_short_name
# python src/data/prepare_evaluation_analogy.py \
#     --analogies-file data/analogies.txt \
#     --output-dir data/analogies/gpt2 \
#     --compact-output True \
#     --schema-uri ../../docs/analogy.schema.json \
#     --device cuda \
#     --model $model_name \
#     --cache_dir $cache_dir 


# Run rationalization task
python src/rationalization/run_analogies.py \
    --rationalization-config config/eva_$FA_name.json \
    --model $model_name \
    --tokenizer $model_name \
    --data-dir data/analogies/$model_short_name/ \
    --importance_results_dir $importance_results \
    --device cuda \
    --logfolder $logfolder_shortname \
    --input_num_ratio 1 \
    --cache_dir $cache_dir



# # for greedy search ---> only once
# # python src/rationalization/migrate_results_analogies.py



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


#  ONLY FOR GPT2
echo $rationale_ratio_for_eva
python src/evaluation/evaluate_analogies.py \
    --importance_results_dir $importance_results \
    --eva_output_dir $eva_output_dir \
    --model $model_name \
    --tokenizer $model_name \
    --logfolder $logfolder_shortname \
    --rationale_size_ratio 0 \
    --rational_size_file "rationalization_results/analogies-greedy-lengths.json" \
    --cache_dir $cache_dir