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
FA_name="ours" # select from all_attention rollout_attention last_attention    
importance_results="rationalization_results/analogies/"$model_short_name"_"$FA_name
echo "importance results (rationalization results is in --->)"
echo importance_results
cache_dir="cache/"


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


#python src/rationalization/migrate_results_analogies.py

eva_output_dir="evaluation_results/analogies/"$model_name"_"$FA_name
mkdir -p $eva_output_dir

echo $rationale_ratio_for_eva
python src/evaluation/evaluate_analogies.py \
    --importance_results_dir $importance_results \
    --eva_output_dir $eva_output_dir \
    --model $model_name \
    --tokenizer $model_name \
    --logfolder "logs/analogies/"$model_name"_"$FA_name \
    --rational_size_ratio 0 \
    --rational_size_file "rationalization_results/analogies-greedy-lengths.json" \
    --cache_dir $cache_dir
