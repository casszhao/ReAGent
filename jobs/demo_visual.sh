#!/bin/bash
#SBATCH --comment=gpt2
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=222G
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

#"facebook/galactica-6.7b"
#"facebook/galactica-1.3b"
#"facebook/galactica-125m"

#"bigscience/bloom-7b1"
#"bigscience/bloomz-1b1"
#"bigscience/bloom-560m"

#"KoboldAI/OPT-6.7B-Erebus"

#"gpt2-medium"
#"gpt2-large"

#### NO
#"stabilityai/FreeWilly2"
#"TheBloke/Llama-2-13B-GPTQ"
#"tiiuae/falcon-40b"
#"mosaicml/mpt-7b-8k-instruct"



# Run rationalization task
python src/rationalization/notebook.py --model_name $model_name