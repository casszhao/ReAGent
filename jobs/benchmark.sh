#!/bin/bash
#SBATCH --comment=benchmark
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --output=jobs.out/%j.log
#SBATCH --time=4-00:00:00

#SBATCH --job-name=benchmark

# Load modules & activate env

module load Anaconda3/2022.10
module load CUDA/11.8.0
source activate dev-inseq     

bash runs/benchmark.sh
