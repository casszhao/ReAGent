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
model_name="gpt2-medium"
model_short_name="gpt2"
hyper="/top3_replace0.1_max3000_batch5"


##########  selecting FA
# select: ours
# select from: all_attention rollout_attention last_attention   
# select from: norm integrated signed

for FA_name in "all_attention" "greedy" "inseq_ig" "last_attention" "norm" "rollout_attention" 
do
importance_results="rationalization_results/analogies/"$model_short_name"_"$FA_name
eva_output_dir="evaluation_results/analogies/"$model_short_name"_"$FA_name
mkdir -p $importance_results
mkdir -p $eva_output_dir
mkdir -p logs/analogies/$model_name"_"$FA_name
logfolder=logs/analogies/$model_name"_"$FA_name
mkdir -p logs/analogies/$model_short_name"_"$FA_name
logfolder_shortname=logs/analogies/$model_short_name"_"$FA_name

    
### evaluate ANT and Ratio
python src/evaluation/evaluate_analogies-old.py \
    --data-dir data/analogies/$model_short_name \
    --target_dir rationalization_results/analogies/"$model_short_name"_inseq_ig \
    --baseline_dir rationalization_results/analogies/gpt2_$FA_name \
    --output-path evaluation_results/analogies/"$model_short_name"_$FA_name/
done










## for different folder name for ours methods 
FA_name="ours" 
importance_results="rationalization_results/analogies/"$model_short_name"_"$FA_name$hyper
eva_output_dir="evaluation_results/analogies/"$model_short_name"_"$FA_name$hyper
mkdir -p $importance_results
mkdir -p $eva_output_dir
mkdir -p logs/analogies/$model_name"_"$FA_name$hyper
logfolder=logs/analogies/$model_name"_"$FA_name$hyper
mkdir -p logs/analogies/$model_short_name"_"$FA_name$hyper
logfolder_shortname=logs/analogies/$model_short_name"_"$FA_name$hyper

    
### evaluate ANT and Ratio
python src/evaluation/evaluate_analogies-old.py \
    --data-dir data/analogies/$model_short_name \
    --target_dir rationalization_results/analogies/"$model_short_name"_inseq_ig \
    --baseline_dir rationalization_results/analogies/gpt2_$FA_name \
    --output-path evaluation_results/analogies/"$model_short_name"_$FA_name/



# for rationale_ratio_for_eva in 0.05 0.1 0.2 0.3 1
# do
# echo "  for rationale "
# echo $rationale_ratio_for_eva
# python src/evaluation/evaluate_analogies.py \
#     --importance_results_dir $importance_results \
#     --eva_output_dir $eva_output_dir \
#     --model $model_name \
#     --tokenizer $model_name \
#     --logfolder $logfolder_shortname \
#     --rationale_size_ratio $rationale_ratio_for_eva \
#     --cache_dir $cache_dir
# done


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