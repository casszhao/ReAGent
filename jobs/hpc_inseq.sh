#!/bin/bash
#SBATCH --comment=basel_FA
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=12
#SBATCH --output=jobs.out/%j.log
#SBATCH --time=4-00:00:00
#SBATCH --mail-user=zhixue.zhao@sheffield.ac.uk

#SBATCH --job-name=optBASE

#$ -m abe


module load Anaconda3/2022.10
module load CUDA/11.7.0
module load libs/boost/1.64.0/gcc-8.2-cmake-3.17.1
# cuDNN/8.0.4.30-CUDA-11.1.1
source activate inseq      
# source .venv/bin/activate           # via venv

model_name="gpt2-medium"   # "gpt2-medium"   "KoboldAI/OPT-6.7B-Erebus"
model_short_name="gpt2"  #"gpt2"    "OPT6B"
cache_dir="cache/"    




FA_name="inseq_ig" 
# select from : all_attention rollout_attention last_attention
# select from: integrated norm signed
importance_results="rationalization_results/analogies/"$model_short_name"_"$FA_name
eva_output_dir="evaluation_results/analogies/"$model_short_name"_"$FA_name
config_file="config/eva_"$FA_name".json"


# # Run rationalization task, get importance distribution
# mkdir -p $importance_results
# mkdir -p $logpath


python src/rationalization/run_analogies.py \
    --rationalization-config config/eva_inseq_ig.json \
    --model $model_name \
    --tokenizer $model_name \
    --data-dir data/analogies/$model_short_name/ \
    --importance_results_dir $importance_results \
    --device cuda \
    --logfolder "logs/analogies/"$model_short_name"_"$FA_name \
    --cache_dir $cache_dir 






## evvaluate different length and soft suff/comp
for rationale_ratio_for_eva in 0.05 0.1 0.2 0.3 1
do
echo "  for rationale "
echo $rationale_ratio_for_eva
python src/evaluation/evaluate_analogies.py \
    --importance_results_dir $importance_results \
    --eva_output_dir $eva_output_dir \
    --model $model_name \
    --tokenizer $model_name \
    --logfolder "logs/analogies/"$model_short_name"_"$FA_name \
    --rational_size_ratio $rationale_ratio_for_eva \
    --cache_dir $cache_dir 
done



### evaluate flexi length
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
