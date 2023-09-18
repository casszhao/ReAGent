
run_one () {
    # tokenizer-name=$1
    # data-name=$2
    # setup-name=$3

    output_dir=plots/benchmark/$3/$2
    mkdir -p $output_dir

    python plot_raw.py \
      --tokenizer-name $1 \
      --source-data-path data/benchmark/$2.txt \
      --result-raw-dir evaluation_results/benchmark/$3/$2  \
      --output-dir $output_dir && (
        echo python script finished $1 $2 $3;
    ) || (
        echo python script error $1 $2 $3;
        exit 1;
    )
}

run_one gpt2-medium tellmewhy2 gpt2_attention
# run_one gpt2-medium tellmewhy2 gpt2_attention_last
# run_one gpt2-medium tellmewhy2 gpt2_attention_rollout
# run_one gpt2-medium tellmewhy2 gpt2_gradient_shap
# run_one gpt2-medium tellmewhy2 gpt2_input_x_gradient
# run_one gpt2-medium tellmewhy2 gpt2_intergrated_gradient
# run_one gpt2-medium tellmewhy2 gpt2_lime
# run_one gpt2-medium tellmewhy2 gpt2_norm
# run_one gpt2-medium tellmewhy2 gpt2_ours

# run_one gpt2-xl tellmewhy2 gpt2_xl_attention
# run_one gpt2-xl tellmewhy2 gpt2_xl_attention_last
# run_one gpt2-xl tellmewhy2 gpt2_xl_attention_rollout
# run_one gpt2-xl tellmewhy2 gpt2_xl_gradient_shap
# run_one gpt2-xl tellmewhy2 gpt2_xl_input_x_gradient
# run_one gpt2-xl tellmewhy2 gpt2_xl_intergrated_gradient
# run_one gpt2-xl tellmewhy2 gpt2_xl_lime
# run_one gpt2-xl tellmewhy2 gpt2_xl_norm
# run_one gpt2-xl tellmewhy2 gpt2_xl_ours

# run_one EleutherAI/gpt-j-6b tellmewhy2 gpt6b_attention
# run_one EleutherAI/gpt-j-6b tellmewhy2 gpt6b_attention_last
# run_one EleutherAI/gpt-j-6b tellmewhy2 gpt6b_attention_rollout
# run_one EleutherAI/gpt-j-6b tellmewhy2 gpt6b_gradient_shap
# run_one EleutherAI/gpt-j-6b tellmewhy2 gpt6b_input_x_gradient
# run_one EleutherAI/gpt-j-6b tellmewhy2 gpt6b_intergrated_gradient
# run_one EleutherAI/gpt-j-6b tellmewhy2 gpt6b_lime
# run_one EleutherAI/gpt-j-6b tellmewhy2 gpt6b_norm
# run_one EleutherAI/gpt-j-6b tellmewhy2 gpt6b_ours

# run_one facebook/opt-350m tellmewhy2 OPT350M_attention
# run_one facebook/opt-350m tellmewhy2 OPT350M_attention_last
# run_one facebook/opt-350m tellmewhy2 OPT350M_attention_rollout
# run_one facebook/opt-350m tellmewhy2 OPT350M_gradient_shap
# run_one facebook/opt-350m tellmewhy2 OPT350M_input_x_gradient
# run_one facebook/opt-350m tellmewhy2 OPT350M_intergrated_gradient
# run_one facebook/opt-350m tellmewhy2 OPT350M_lime
# run_one facebook/opt-350m tellmewhy2 OPT350M_norm
# run_one facebook/opt-350m tellmewhy2 OPT350M_ours

# run_one facebook/opt-1.3b tellmewhy2 OPT1B_attention
# run_one facebook/opt-1.3b tellmewhy2 OPT1B_attention_last
# run_one facebook/opt-1.3b tellmewhy2 OPT1B_attention_rollout
# run_one facebook/opt-1.3b tellmewhy2 OPT1B_gradient_shap
# run_one facebook/opt-1.3b tellmewhy2 OPT1B_input_x_gradient
# run_one facebook/opt-1.3b tellmewhy2 OPT1B_intergrated_gradient
# run_one facebook/opt-1.3b tellmewhy2 OPT1B_lime
# run_one facebook/opt-1.3b tellmewhy2 OPT1B_norm
# run_one facebook/opt-1.3b tellmewhy2 OPT1B_ours

# run_one KoboldAI/OPT-6.7B-Erebus tellmewhy2 OPT6B_attention
# run_one KoboldAI/OPT-6.7B-Erebus tellmewhy2 OPT6B_attention_last
# run_one KoboldAI/OPT-6.7B-Erebus tellmewhy2 OPT6B_attention_rollout
# run_one KoboldAI/OPT-6.7B-Erebus tellmewhy2 OPT6B_gradient_shap
# run_one KoboldAI/OPT-6.7B-Erebus tellmewhy2 OPT6B_input_x_gradient
# run_one KoboldAI/OPT-6.7B-Erebus tellmewhy2 OPT6B_intergrated_gradient
# run_one KoboldAI/OPT-6.7B-Erebus tellmewhy2 OPT6B_lime
# run_one KoboldAI/OPT-6.7B-Erebus tellmewhy2 OPT6B_norm
# run_one KoboldAI/OPT-6.7B-Erebus tellmewhy2 OPT6B_ours
