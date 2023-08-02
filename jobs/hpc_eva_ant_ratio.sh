#!/bin/bash
#SBATCH --comment=ant_ratio
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=jobs.out/%j.log
#SBATCH --time=4-00:00:00
#SBATCH --mail-user=zhixue.zhao@sheffield.ac.uk

#$ -N ant_ratio
#$ -m abe



module load Anaconda3/2022.10
module load CUDA/11.8.0
source activate dev-inseq     

cache_dir="cache/"
model_name="gpt2-medium"  # KoboldAI/OPT-6.7B-Erebus
model_short_name="gpt2"
hyper="/top3_replace0.1_max3000_batch5"


##########  selecting FA
# select: ours
# select from: all_attention rollout_attention last_attention   
# select from: norm integrated signed

#for FA_name in "all_attention" "inseq_ig" "last_attention" "norm" "rollout_attention" "signed" "ours/top3_replace0.3_max3000_batch3" 
for FA_name in "ours/top3_replace0.3_max3000_batch5" "ours/top3_replace0.3_max3000_batch8" "ours/top3_replace0.3_max5000_batch8" "ours/top5_replace0.3_max5000_batch8"
# do
# #FA_name="all_attention"
# for rational_size in 3 5 7 10
# do
# importance_results="rationalization_results/analogies/"$model_short_name"_"$FA_name
# eva_output_dir="evaluation_results/analogies/"$model_short_name"_"$FA_name
# mkdir -p $importance_results
# mkdir -p $eva_output_dir
# mkdir -p logs/analogies/$model_name"_"$FA_name
# logfolder=logs/analogies/$model_name"_"$FA_name
# mkdir -p logs/analogies/$model_short_name"_"$FA_name
# logfolder_shortname=logs/analogies/$model_short_name"_"$FA_name


# ### evaluate ANT and Ratio
# python src/evaluation/evaluate_analogies-old.py \
#     --data-dir data/analogies/$model_short_name \
#     --target_dir rationalization_results/analogies/"$model_short_name"_$FA_name \
#     --baseline_dir rationalization_results/analogies/gpt2_$FA_name \
#     --model_short_name $model_short_name \
#     --fa_name $FA_name \
#     --rational_size_override $rational_size
# echo "done for"
# echo $rational_size $FA_name
# done

python src/evaluation/evaluate_analogies-old.py \
    --data-dir data/analogies/$model_short_name \
    --target_dir rationalization_results/analogies/"$model_short_name"_$FA_name \
    --baseline_dir rationalization_results/analogies/gpt2_$FA_name \
    --model_short_name $model_short_name \
    --fa_name $FA_name \
    --rational_size_override $rational_size


done
