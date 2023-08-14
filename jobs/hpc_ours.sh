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

#SBATCH --job-name=OPT350M
#$ -m abe
export TRANSFORMERS_CACHE=/mnt/parscratch/users/cass/seq_rationales/cache/

# Load modules & activate env

module load Anaconda3/2022.10
module load CUDA/11.8.0
source activate dev-inseq     

cache_dir="cache/"

model_name="facebook/opt-350m"
# "gpt2-medium"
# "gpt2-xl"
# "EleutherAI/gpt-j-6b"
# "facebook/opt-350m"
# "facebook/opt-1.3b"
# "KoboldAI/OPT-6.7B-Erebus"
model_short_name="OPT350M" 
# gpt2 gpt2_xl gpt6b
# OPT350M OPT1B OPT6B

# hyper="/top3_replace0.1_max3000_batch5"
hyper="/top3_replace0.1_max5000_batch5"


##########  selecting FA
# select: ours
# select from: all_attention attention_rollout attention_last   
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
#     --output-dir data/analogies/$model_short_name \
#     --compact-output True \
#     --schema-uri ../../docs/analogy.schema.json \
#     --device cuda \
#     --model $model_name \
#     --cache_dir $cache_dir 


# # Run rationalization task
# python src/rationalization/run_analogies.py \
#     --rationalization-config config/$hyper.json \
#     --model $model_name \
#     --tokenizer $model_name \
#     --data-dir data/analogies/$model_short_name \
#     --importance_results_dir $importance_results \
#     --device cuda \
#     --logfolder $logfolder_shortname \
#     --input_num_ratio 1 \
#     --cache_dir $cache_dir



python src/evaluation/evaluate_analogies.py \
    --importance_results_dir $importance_results \
    --eva_output_dir $eva_output_dir \
    --model $model_name \
    --tokenizer $model_name \
    --logfolder $logfolder_shortname \
    --rationale_size_ratio 1 \
    --cache_dir $cache_dir




# # for greedy search ---> only once
# # python src/rationalization/migrate_results_analogies.py


# #  ONLY FOR GPT2
# echo $rationale_ratio_for_eva
# python src/evaluation/evaluate_analogies.py \
#     --importance_results_dir $importance_results \
#     --eva_output_dir $eva_output_dir \
#     --model $model_name \
#     --tokenizer $model_name \
#     --logfolder $logfolder_shortname \
#     --rationale_size_ratio 0 \
#     --rational_size_file "rationalization_results/analogies-greedy-lengths.json" \
#     --cache_dir $cache_dir