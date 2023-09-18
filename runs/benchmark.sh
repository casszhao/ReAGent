
run_one () {
    # model=$1
    # model_shortname=$2
    # testing_data_name=$3
    # method=$4
    
    echo python script start $1 $2 $3 $4;
    python src/sequence_rationalization.py --model $1 --model_shortname $2 --testing_data_name $3 --method $4 --if_image True && (
        echo python script finished $1 $2 $3 $4;
    ) || (
        echo python script error $1 $2 $3 $4;
        exit 1;
    )
}

# run_one gpt2-medium gpt2 wikitext attention
# run_one gpt2-medium gpt2 wikitext attention_last
# run_one gpt2-medium gpt2 wikitext attention_rollout
# run_one gpt2-medium gpt2 wikitext gradient_shap
# run_one gpt2-medium gpt2 wikitext input_x_gradient
# run_one gpt2-medium gpt2 wikitext integrated_gradients
# run_one gpt2-medium gpt2 wikitext norm
# run_one gpt2-medium gpt2 wikitext lime
# run_one gpt2-medium gpt2 wikitext ours
# run_one gpt2-medium gpt2 tellmewhy2 attention
# run_one gpt2-medium gpt2 tellmewhy2 attention_last
# run_one gpt2-medium gpt2 tellmewhy2 attention_rollout
# run_one gpt2-medium gpt2 tellmewhy2 gradient_shap
# run_one gpt2-medium gpt2 tellmewhy2 input_x_gradient
# run_one gpt2-medium gpt2 tellmewhy2 integrated_gradients
# run_one gpt2-medium gpt2 tellmewhy2 norm
# run_one gpt2-medium gpt2 tellmewhy2 lime
# run_one gpt2-medium gpt2 tellmewhy2 ours

# run_one gpt2-xl gpt2_xl wikitext attention
# run_one gpt2-xl gpt2_xl wikitext attention_last
# run_one gpt2-xl gpt2_xl wikitext attention_rollout
# run_one gpt2-xl gpt2_xl wikitext gradient_shap
# run_one gpt2-xl gpt2_xl wikitext input_x_gradient
# run_one gpt2-xl gpt2_xl wikitext integrated_gradients
# run_one gpt2-xl gpt2_xl wikitext norm
# run_one gpt2-xl gpt2_xl wikitext lime
# run_one gpt2-xl gpt2_xl wikitext ours
# run_one gpt2-xl gpt2_xl tellmewhy2 attention
# run_one gpt2-xl gpt2_xl tellmewhy2 attention_last
# run_one gpt2-xl gpt2_xl tellmewhy2 attention_rollout
# run_one gpt2-xl gpt2_xl tellmewhy2 gradient_shap
# run_one gpt2-xl gpt2_xl tellmewhy2 input_x_gradient
# run_one gpt2-xl gpt2_xl tellmewhy2 integrated_gradients
# run_one gpt2-xl gpt2_xl tellmewhy2 norm
# run_one gpt2-xl gpt2_xl tellmewhy2 lime
# run_one gpt2-xl gpt2_xl tellmewhy2 ours

run_one EleutherAI/gpt-j-6b gpt6b wikitext attention
run_one EleutherAI/gpt-j-6b gpt6b wikitext attention_last
run_one EleutherAI/gpt-j-6b gpt6b wikitext attention_rollout
run_one EleutherAI/gpt-j-6b gpt6b wikitext gradient_shap
run_one EleutherAI/gpt-j-6b gpt6b wikitext input_x_gradient
run_one EleutherAI/gpt-j-6b gpt6b wikitext integrated_gradients
run_one EleutherAI/gpt-j-6b gpt6b wikitext norm
run_one EleutherAI/gpt-j-6b gpt6b wikitext lime
run_one EleutherAI/gpt-j-6b gpt6b wikitext ours
run_one EleutherAI/gpt-j-6b gpt6b tellmewhy2 attention
run_one EleutherAI/gpt-j-6b gpt6b tellmewhy2 attention_last
run_one EleutherAI/gpt-j-6b gpt6b tellmewhy2 attention_rollout
run_one EleutherAI/gpt-j-6b gpt6b tellmewhy2 gradient_shap
run_one EleutherAI/gpt-j-6b gpt6b tellmewhy2 input_x_gradient
run_one EleutherAI/gpt-j-6b gpt6b tellmewhy2 integrated_gradients
run_one EleutherAI/gpt-j-6b gpt6b tellmewhy2 norm
run_one EleutherAI/gpt-j-6b gpt6b tellmewhy2 lime
run_one EleutherAI/gpt-j-6b gpt6b tellmewhy2 ours

run_one facebook/opt-350m OPT350M wikitext attention
run_one facebook/opt-350m OPT350M wikitext attention_last
run_one facebook/opt-350m OPT350M wikitext attention_rollout
run_one facebook/opt-350m OPT350M wikitext gradient_shap
run_one facebook/opt-350m OPT350M wikitext input_x_gradient
run_one facebook/opt-350m OPT350M wikitext integrated_gradients
run_one facebook/opt-350m OPT350M wikitext norm
run_one facebook/opt-350m OPT350M wikitext lime
run_one facebook/opt-350m OPT350M wikitext ours
run_one facebook/opt-350m OPT350M tellmewhy2 attention
run_one facebook/opt-350m OPT350M tellmewhy2 attention_last
run_one facebook/opt-350m OPT350M tellmewhy2 attention_rollout
run_one facebook/opt-350m OPT350M tellmewhy2 gradient_shap
run_one facebook/opt-350m OPT350M tellmewhy2 input_x_gradient
run_one facebook/opt-350m OPT350M tellmewhy2 integrated_gradients
run_one facebook/opt-350m OPT350M tellmewhy2 norm
run_one facebook/opt-350m OPT350M tellmewhy2 lime
run_one facebook/opt-350m OPT350M tellmewhy2 ours

run_one facebook/opt-1.3b OPT1B wikitext attention
run_one facebook/opt-1.3b OPT1B wikitext attention_last
run_one facebook/opt-1.3b OPT1B wikitext attention_rollout
run_one facebook/opt-1.3b OPT1B wikitext gradient_shap
run_one facebook/opt-1.3b OPT1B wikitext input_x_gradient
run_one facebook/opt-1.3b OPT1B wikitext integrated_gradients
run_one facebook/opt-1.3b OPT1B wikitext norm
run_one facebook/opt-1.3b OPT1B wikitext lime
run_one facebook/opt-1.3b OPT1B wikitext ours
run_one facebook/opt-1.3b OPT1B tellmewhy2 attention
run_one facebook/opt-1.3b OPT1B tellmewhy2 attention_last
run_one facebook/opt-1.3b OPT1B tellmewhy2 attention_rollout
run_one facebook/opt-1.3b OPT1B tellmewhy2 gradient_shap
run_one facebook/opt-1.3b OPT1B tellmewhy2 input_x_gradient
run_one facebook/opt-1.3b OPT1B tellmewhy2 integrated_gradients
run_one facebook/opt-1.3b OPT1B tellmewhy2 norm
run_one facebook/opt-1.3b OPT1B tellmewhy2 lime
run_one facebook/opt-1.3b OPT1B tellmewhy2 ours

run_one KoboldAI/OPT-6.7B-Erebus OPT6B wikitext attention
run_one KoboldAI/OPT-6.7B-Erebus OPT6B wikitext attention_last
run_one KoboldAI/OPT-6.7B-Erebus OPT6B wikitext attention_rollout
run_one KoboldAI/OPT-6.7B-Erebus OPT6B wikitext gradient_shap
run_one KoboldAI/OPT-6.7B-Erebus OPT6B wikitext input_x_gradient
run_one KoboldAI/OPT-6.7B-Erebus OPT6B wikitext integrated_gradients
run_one KoboldAI/OPT-6.7B-Erebus OPT6B wikitext norm
run_one KoboldAI/OPT-6.7B-Erebus OPT6B wikitext lime
run_one KoboldAI/OPT-6.7B-Erebus OPT6B wikitext ours
run_one KoboldAI/OPT-6.7B-Erebus OPT6B tellmewhy2 attention
run_one KoboldAI/OPT-6.7B-Erebus OPT6B tellmewhy2 attention_last
run_one KoboldAI/OPT-6.7B-Erebus OPT6B tellmewhy2 attention_rollout
run_one KoboldAI/OPT-6.7B-Erebus OPT6B tellmewhy2 gradient_shap
run_one KoboldAI/OPT-6.7B-Erebus OPT6B tellmewhy2 input_x_gradient
run_one KoboldAI/OPT-6.7B-Erebus OPT6B tellmewhy2 integrated_gradients
run_one KoboldAI/OPT-6.7B-Erebus OPT6B tellmewhy2 norm
run_one KoboldAI/OPT-6.7B-Erebus OPT6B tellmewhy2 lime
run_one KoboldAI/OPT-6.7B-Erebus OPT6B tellmewhy2 ours
