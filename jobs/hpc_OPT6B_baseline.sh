#!/bin/bash
#SBATCH --comment=opt
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=12
#SBATCH --output=jobs.out/%j.log
#SBATCH --time=4-00:00:00
#SBATCH --mail-user=zhixue.zhao@sheffield.ac.uk

#$ -N opt
#$ -m norm

# Load modules & activate env

module load Anaconda3/2022.10
module load CUDA/11.8.0

# Activate env
source activate dev-inseq      # via conda

cache_dir="cache/"
model_name="KoboldAI/OPT-6.7B-Erebus"
model_short_name="OPT6B"

##########  selecting FA
# select from: all_attention rollout_attention last_attention   
# select from: norm 
FA_name="norm" 
config_file=config/eva_OPT6B_$FA_name.json

importance_results="rationalization_results/analogies/"$model_short_name"_"$FA_name
eva_output_dir="evaluation_results/analogies/"$model_short_name"_"$FA_name
mkdir -p $importance_results
mkdir -p $eva_output_dir
mkdir -p logs/analogies/$model_name"_"$FA_name
logfolder=logs/analogies/$model_name"_"$FA_name
mkdir -p logs/analogies/$model_short_name"_"$FA_name
logfolder_shortname=logs/analogies/$model_short_name"_"$FA_name


#  ## Run rationalization task
python src/rationalization/run_analogies.py \
    --rationalization-config $config_file \
    --model $model_name \
    --tokenizer $model_name \
    --data-dir data/analogies/$model_short_name/ \
    --importance_results_dir $importance_results \
    --device cuda \
    --logfolder $logfolder_shortname \
    --input_num_ratio 1 \
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
    --logfolder $logfolder_shortname \
    --rationale_size_ratio $rationale_ratio_for_eva \
    --cache_dir $cache_dir
done
