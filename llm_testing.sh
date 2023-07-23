#!/bin/bash
#SBATCH --comment=llama
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=12
#SBATCH --output=jobs.out/%j.log
#SBATCH --time=4-00:00:00
#SBATCH --mail-user=zhixue.zhao@sheffield.ac.uk

#$ -N llama
#$ -m abe


# Load modules & activate env

module load Anaconda3/2022.10
module load cuDNN/8.0.4.30-CUDA-11.1.1
source activate seq      # via conda


python src/evaluation/gen_map_rational_size.py #llm_testing.sh