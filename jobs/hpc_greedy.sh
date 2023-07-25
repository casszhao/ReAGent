#!/bin/bash
#SBATCH --comment=gpt2
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=12
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
# source .venv/bin/activate           # via venv

model_name="gpt2-medium"
model_short_name="gpt2"
hyper="/top5_replace0.3_max3000"

FA_name="ours" 
# select from: all_attention rollout_attention last_attention   
# select from: norm integrated signed
importance_results="rationalization_results/analogies/"$model_short_name"_"$FA_name$hyper
cache_dir="cache/"

echo "importance results (rationalization results is in --->)"
echo $importance_results

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


# # Run rationalization task
# mkdir -p "$importance_results"
# mkdir -p "$logpath"
# python src/rationalization/run_analogies.py \
#     --rationalization-config config/aggregation.replacing_delta_prob.postag.json \
#     --model $model_name \
#     --tokenizer $model_name \
#     --data-dir data/analogies/$model_short_name/ \
#     --importance_results_dir $importance_results \
#     --device cuda \
#     --logfolder "logs/analogies/"$model_short_name"_"$FA_name \
#     --input_num_ratio 1 \
#     --cache_dir $cache_dir



eva_output_dir="evaluation_results/analogies/"$model_name"_"$FA_name$hyper
mkdir -p $eva_output_dir


echo $rationale_ratio_for_eva
python src/evaluation/evaluate_analogies.py \
    --importance_results_dir $importance_results \
    --eva_output_dir $eva_output_dir \
    --model $model_name \
    --tokenizer $model_name \
    --logfolder "logs/analogies/"$model_name"_"$FA_name$hyper \
    --rational_size_ratio 0 \
    --rational_size_file "rationalization_results/analogies-greedy-lengths.json" \
    --cache_dir $cache_dir



for rationale_ratio_for_eva in 0.05 0.1 0.2 0.3 1
do
echo "  for rationale "
echo $rationale_ratio_for_eva
python src/evaluation/evaluate_analogies.py \
    --importance_results_dir $importance_results \
    --eva_output_dir $eva_output_dir \
    --model $model_name \
    --tokenizer $model_name \
    --logfolder "logs/analogies/"$model_name"_"$FA_name$hyper \
    --rational_size_ratio $rationale_ratio_for_eva \
    --rational_size_file None \
    --cache_dir $cache_dir
done