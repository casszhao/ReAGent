#!/bin/bash
#SBATCH --comment=basel_FA
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=12
#SBATCH --output=jobs.out/%j.log
#SBATCH --time=4-00:00:00
#SBATCH --mail-user=zhixue.zhao@sheffield.ac.uk

#SBATCH --job-name=gpt2_xl

#$ -m abe

export TRANSFORMERS_CACHE=/mnt/parscratch/users/cass/seq_rationales/cache/

module load Anaconda3/2022.10
module load CUDA/11.8.0
source activate dev-inseq      
# source .venv/bin/activate           # via venv


model_name="KoboldAI/OPT-6.7B-Erebus"
# "gpt2-medium"
# "gpt2-xl"
# "EleutherAI/gpt-j-6b"
# "facebook/opt-350m"
# "facebook/opt-1.3b"
# "KoboldAI/OPT-6.7B-Erebus"
model_short_name="OPT6B"
# gpt2 gpt2_xl gpt6b
# OPT350M OPT1B OPT6B

cache_dir="./cache/"    

FA_name='lime'



# FA_name="integrated_gradients"
# for FA_name in "attention_rollout" "attention_last" "attention" "gradient_shap" "integrated_gradients" "input_x_gradient"
# do
# select from: "attention_rollout" "attention_last" "attention"
# select from: "norm" "gradient_shap" "integrated_gradients" "input_x_gradient" 
importance_results="rationalization_results/analogies/"$model_short_name"_"$FA_name
eva_output_dir="evaluation_results/analogies/"$model_short_name"_"$FA_name
config_file="config/eva_"$FA_name".json"


# # Run rationalization task, get importance distribution
# mkdir -p $importance_results
# mkdir -p $logpath


python src/rationalization/run_analogies.py \
    --rationalization-config config/eva_$FA_name.json \
    --model $model_name \
    --tokenizer $model_name \
    --data-dir data/analogies/$model_short_name/ \
    --importance_results_dir $importance_results \
    --device cuda \
    --logfolder "logs/analogies/"$model_short_name"_"$FA_name \
    --cache_dir $cache_dir 


## evvaluate different length and soft suff/comp
rationale_ratio_for_eva=1

echo "  for rationale "
echo $rationale_ratio_for_eva
python src/evaluation/evaluate_analogies.py \
    --importance_results_dir $importance_results \
    --eva_output_dir $eva_output_dir \
    --model $model_name \
    --tokenizer $model_name \
    --logfolder "logs/analogies/"$model_short_name"_"$FA_name \
    --rationale_size_ratio $rationale_ratio_for_eva \
    --cache_dir $cache_dir 


#done
# ### evaluate flexi length
# echo $rationale_ratio_for_eva
# python src/evaluation/evaluate_analogies.py \
#     --importance_results_dir $importance_results \
#     --eva_output_dir $eva_output_dir \
#     --model $model_name \
#     --tokenizer $model_name \
#     --logfolder "logs/analogies/"$model_name"_"$FA_name$hyper \
#     --rationale_size_ratio 0 \
#     --rational_size_file "rationalization_results/analogies-greedy-lengths.json" \
#     --cache_dir $cache_dir




# # hyper="/top3_replace0.1_max3000_batch5"
# hyper="/top3_replace0.1_max5000_batch5"


# ##########  selecting FA
# # select: ours
# # select from: all_attention attention_rollout attention_last   
# # select from: norm integrated signed
# FA_name="ours"

# importance_results="rationalization_results/analogies/"$model_short_name"_"$FA_name$hyper
# eva_output_dir="evaluation_results/analogies/"$model_short_name"_"$FA_name$hyper
# mkdir -p $importance_results
# mkdir -p $eva_output_dir
# mkdir -p logs/analogies/$model_name"_"$FA_name$hyper
# logfolder=logs/analogies/$model_name"_"$FA_name$hyper
# mkdir -p logs/analogies/$model_short_name"_"$FA_name$hyper
# logfolder_shortname=logs/analogies/$model_short_name"_"$FA_name$hyper


# python src/evaluation/evaluate_analogies.py \
#     --importance_results_dir $importance_results \
#     --eva_output_dir $eva_output_dir \
#     --model $model_name \
#     --tokenizer $model_name \
#     --logfolder $logfolder_shortname \
#     --rationale_size_ratio 1 \
#     --cache_dir $cache_dir


