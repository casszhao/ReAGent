#!/bin/bash
#SBATCH --comment=gpt2
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=222G
#SBATCH --output=jobs.out/%j.log
#SBATCH --time=4-00:00:00
#SBATCH --mail-user=zhixue.zhao@sheffield.ac.uk

#$ -N gpt2
#$ -m abe


# Load modules & activate env

module load Anaconda3/2022.10
module load cuDNN/8.0.4.30-CUDA-11.1.1

# Activate env
source activate seq      # via conda

cache_dir="cache/"
model_name="gpt2-medium"
model_short_name="gpt2"
hyper="/top5_replace0.3_max5000_batch8"

##########  selecting FA
# select: ours
# select from: all_attention rollout_attention last_attention   
# select from: norm integrated signed
FA_name="ours" 

importance_results="rationalization_results/analogies/"$model_short_name"_"$FA_name$hyper
eva_output_dir="evaluation_results/analogies/"$model_short_name"_"$FA_name$hyper
mkdir -p $importance_results
mkdir -p $eva_output_dir
mkdir -p logs/analogies/$model_name"_"$FA_name$hyper
logfolder=logs/analogies/$model_name"_"$FA_name$hyper
mkdir -p logs/analogies/$model_short_name"_"$FA_name$hyper
logfolder_shortname=logs/analogies/$model_short_name"_"$FA_name$hyper


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
python src/rationalization/notebook.py