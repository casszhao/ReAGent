#!/bin/bash
#SBATCH --comment=gpt2
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --output=jobs.out/%j.log
#SBATCH --time=4-00:00:00
#SBATCH --mail-user=zhixue.zhao@sheffield.ac.uk

#SBATCH --job-name=gpt2_xl

#$ -m abe

export TRANSFORMERS_CACHE=/mnt/parscratch/users/cass/seq_rationales/cache/

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


FA_name="integrated_gradients"
# "integrated_gradients" "input_x_gradient" "attention" "attention_rollout" "attention_last" 

# for FA_name in "ours" "norm" "gradient_shap" "integrated_gradients" "input_x_gradient" "attention" "attention_rollout" "attention_last" 
# ##########  selecting FA "ours" "norm"
# # "norm" "gradient_shap" "integrated_gradients" "input_x_gradient" 
# #  "attention" "attention_rollout" "attention_last" 
# #   "ours" 
# do
for dataset in 'wikitext' 'tellmewhy' 
do
python src/benchmark.py \
    --model $model_name \
    --model_shortname $model_short_name \
    --method $FA_name \
    --stride 2 \
    --max_new_tokens 10 \
    --cache_dir $cache_dir \
    --testing_data_name $dataset
done
# done
