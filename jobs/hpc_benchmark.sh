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

#$ -N gpt2
#$ -m abe


# Load modules & activate env

module load Anaconda3/2022.10
module load CUDA/11.8.0

# Activate env
source activate dev-inseq      # via conda

cache_dir="cache/"
model_name="EleutherAI/gpt-j-6b"
# gpt2-medium, gpt2-xl, EleutherAI/gpt-j-6b, 
# facebook/opt-350m, facebook/opt-1.3b, and KoboldAI/OPT-6.7B-Erebus
model_short_name="gpt6b" 
# gpt2 gpt2_xl gpt6b
# OPT350M OPT1B OPT6B



##########  selecting FA
# select: ours
# select from: all_attention rollout_attention last_attention   
# select from: norm integrated signed




for FA_name in "rollout_attention" "rollout_attention" "norm"
# "gradient_shap" "integrated_gradients" "input_x_gradient" "attention" "ours" 
do
python src/benchmark.py \
    --model $model_name \
    --model_shortname $model_short_name \
    --method $FA_name \
    --stride 2 \
    --max_new_tokens 10 \
    --cache_dir $cache_dir \
    --testing_data_name "wikitext"
done
