
module load Anaconda3/2022.10
module load CUDA/11.8.0

# Activate env
source activate dev-inseq      # via conda

cache_dir="cache/"
model_name="KoboldAI/OPT-6.7B-Erebus"
model_short_name="OPT6B" 

FA_name="attention_last"
python src/benchmark.py \
    --model $model_name \
    --model_shortname $model_short_name \
    --method $FA_name \
    --stride 2 \
    --max_new_tokens 10 \
    --cache_dir $cache_dir \
    --testing_data_name "tellmewhy"

