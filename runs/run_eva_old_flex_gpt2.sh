#!/bin/bash

methods=(
    all_attention
    greedy
    inseq_ig
    integrated
    last_attention
    norm
    ours/top3_replace0.3_max3000_batch3
    ours/top3_replace0.1_max3000_batch5
    rollout_attention
    signed
)

for method in "${methods[@]}"; do
    echo "Run with method: $method"
    python src/evaluation/evaluate_analogies-old.py \
        --data-dir data/analogies/gpt2 \
        --target_dir rationalization_results/analogies/gpt2_$method \
        --output_path evaluation_results/analogies/gpt2_$method/0_ante_nod_flex.csv \
        --baseline_dir rationalization_results/analogies/gpt2_exhaustive \
        --rational_size_file rationalization_results/analogies-greedy-lengths.json
done

echo "All done!"
