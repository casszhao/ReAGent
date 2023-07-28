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
#$ -m abe


# Load modules & activate env

module load Anaconda3/2022.10
module load CUDA/11.8.0

# Activate env
source activate dev-inseq      # via conda

cache_dir="cache/"
model_name="gpt2-medium"
model_short_name="gpt2"
# hyper="/top3_replace0.1_max3000_batch5"
hyper="/top3_replace0.3_max3000_batch8"


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
# python src/rationalization/run_analogies.py \
#     --rationalization-config config/$model_short_name$hyper".json" \
#     --model $model_name \
#     --tokenizer $model_name \
#     --data-dir data/analogies/$model_short_name/ \
#     --importance_results_dir $importance_results \
#     --device cuda \
#     --logfolder $logfolder_shortname \
#     --input_num_ratio 1 \
#     --cache_dir $cache_dir



# # for greedy search ---> only once
# # python src/rationalization/migrate_results_analogies.py



# python src/evaluation/evaluate_analogies-old.py \
#     --data-dir "data/analogies/"$model_short_name \
#     --target_dir $importance_results \
#     --output-path $eva_output_dir \
#     --baseline_dir $importance_results



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