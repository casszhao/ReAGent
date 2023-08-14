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

bash runs/run_analogies_opt_inseq_gradient_shap.sh
bash runs/eva_analohies_opt_inseq_gradient_shap.sh
