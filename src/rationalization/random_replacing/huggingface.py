import torch
from importance_score_evaluator import ImportanceScoreEvaluator
from token_replacement.token_sampler.inferential import InferentialTokenSampler
from rationalizer import Rationalizer
from stopping_condition_evaluator.top_k import TopKStoppingConditionEvaluator
from token_replacement.token_sampler.uniform import UniformTokenSampler
from token_replacement.token_replacer.uniform import UniformTokenReplacer

@torch.no_grad()
def main():

    from transformers import AutoTokenizer, AutoModelWithLMHead

    # ======== model loading ========

    # Load model from Hugging Face
    model = AutoModelWithLMHead.from_pretrained("gpt2-medium")
    tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")

    model.cuda()
    model.eval()
    
    # ======== prepare data ========

    # batch with size 1
    input_string = [ 
        "I love eating breakfast out the"
    ]

    # generate prediction 
    input_ids = tokenizer(input_string, return_tensors='pt')['input_ids'].to(model.device)
    generated_input = model.generate(input_ids=input_ids, max_length=8, do_sample=False) 
    print(' generated input -->', [ [ tokenizer.decode(token) for token in seq] for seq in generated_input ])

    # extract target from prediction
    target_id = generated_input[:, input_ids.shape[1]]
    print(' target -->', [ tokenizer.decode(token) for token in target_id ])

    # ======== hyper-parameters ========

    # replacing ratio during importance score updating
    replacing_ratio = 0.5
    # keep top n word based on importance score for both stop condition evaluation and rationalization
    top_n_ratio = 0.5
    # stop when target exist in top k predictions
    top_k = 0.5

    # ======== rationalization ========
    
    approach_sample_replacing_token = "uniform"
    # approach_sample_replacing_token = "inference"

    # prepare rationalizer
    if approach_sample_replacing_token == "uniform":
        # Approach 1: sample replacing token from uniform distribution
        rationalizer = Rationalizer(
            importance_score_evaluator=ImportanceScoreEvaluator(
                model=model, 
                tokenizer=tokenizer, 
                token_replacer=UniformTokenReplacer(
                    token_sampler=UniformTokenSampler(tokenizer), 
                    ratio=replacing_ratio
                ),
                stopping_condition_evaluator=TopKStoppingConditionEvaluator(
                    model=model, 
                    token_sampler=UniformTokenSampler(tokenizer), 
                    top_k=3, 
                    top_n_ratio=top_n_ratio, 
                    tokenizer=tokenizer
                )
            ), 
            top_n_ratio=top_n_ratio
        )
    elif approach_sample_replacing_token == "inference":
        # Approach 2: sample replacing token from model inference
        rationalizer = Rationalizer(
            importance_score_evaluator=ImportanceScoreEvaluator(
                model=model, 
                tokenizer=tokenizer, 
                token_replacer=UniformTokenReplacer(
                    token_sampler=InferentialTokenSampler(tokenizer=tokenizer, model=model), 
                    ratio=replacing_ratio
                ),
                stopping_condition_evaluator=TopKStoppingConditionEvaluator(
                    model=model, 
                    token_sampler=InferentialTokenSampler(tokenizer=tokenizer, model=model), 
                    top_k=3, 
                    top_n_ratio=top_n_ratio, 
                    tokenizer=tokenizer
                )
            ), 
            top_n_ratio=top_n_ratio
        )
    else:
        raise ValueError("Invalid approach_sample_replacing_token")
    
    # rationalization
    pos_rational = rationalizer.rationalize(input_ids, generated_input[:, input_ids.shape[1]])

    # convert result (for 1st sequence in the batch)
    ids_rational = input_ids[0, pos_rational[0]]
    text_rational = [ tokenizer.decode([id_rational]) for id_rational in ids_rational ]

    print()
    print(f"========================")
    print()
    print(f'Input --> {input_string[0]}')
    print(f'Target --> {tokenizer.decode(target_id[0])}')
    print(f"Rational positions --> {pos_rational}")
    print(f"Rational words --> {text_rational}")

if __name__ == '__main__':
    main()
