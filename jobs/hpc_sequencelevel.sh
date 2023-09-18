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

#SBATCH --job-name=lime-OPT1B

#$ -m abe

export TRANSFORMERS_CACHE="/mnt/parscratch/users/cass/seq_rationales/cache/"

# Load modules & activate env

module load Anaconda3/2022.10
module load CUDA/11.8.0

# Activate env
source activate dev-inseq      # via conda

cache_dir="cache/"
model_name="facebook/opt-1.3b"
# "gpt2-medium"
# "gpt2-xl"
# "EleutherAI/gpt-j-6b"
# "facebook/opt-350m"
# "facebook/opt-1.3b"
# "KoboldAI/OPT-6.7B-Erebus"
model_short_name="OPT1B" 
# gpt2 gpt2_xl gpt6b
# OPT350M OPT1B OPT6B

dataset='tellmewhy2'
FA_name="input_x_gradient"


# for FA_name in "ours" "gradient_shap" "integrated_gradients" "input_x_gradient" "attention" "attention_rollout" "attention_last" 
# ##########  selecting FA "ours" "norm"
# # "norm" "gradient_shap" "integrated_gradients" "input_x_gradient" 
# #  "attention" "attention_rollout" "attention_last" 
# #   "ours" 
# do
# for dataset in 'wikitext' 'tellmewhy2' 'wikitext2'
# do
#dataset='wikitext'
python src/sequence_rationalization.py \
    --model $model_name \
    --model_shortname $model_short_name \
    --method $FA_name \
    --stride 1 \
    --max_new_tokens 10 \
    --cache_dir $cache_dir \
    --testing_data_name $dataset \
    --if_image True
#done
# done
