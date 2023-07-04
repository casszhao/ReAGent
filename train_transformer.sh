#!/bin/bash
<<<<<<< HEAD
=======
#SBATCH --comment=train_transformers_translation
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=output.%j.test.out
#SBATCH --time=4-00:00:00
#SBATCH --mail-user=zhixue.zhao@sheffield.ac.uk

# Load modules & activate env

module load Anaconda3/2022.10
module load cuDNN/8.0.4.30-CUDA-11.1.1

# Activate env
source activate seq      # via conda
>>>>>>> 1a4e776116a4f16f703012239fe61b8c8fa0491f


CHECKPOINT_DIR="./model/"
fairseq-train \
    data-bin/iwslt14.tokenized.de-en \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --encoder-ffn-embed-dim 1024 --decoder-ffn-embed-dim 1024 \
    --encoder-attention-heads 4 --decoder-attention-heads 4 \
    --max-tokens 4096 \
    --max-update 75000 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --tensorboard-logdir=iwslt_logs/standard_iwslt \
    --save-dir $CHECKPOINT_DIR/standard_iwslt \
    --no-epoch-checkpoints \
    --fp16 --word-dropout-mixture 0. 


fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --path $CHECKPOINT_DIR/standard_iwslt/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe