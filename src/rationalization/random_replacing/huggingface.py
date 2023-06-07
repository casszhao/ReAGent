import torch

from token_replacement.token_sampler.uniform import UniformTokenSampler
from token_replacement.token_replacer.uniform import UniformTokenReplacer
from token_replacement.token_replacer.ranking import RankingTokenReplacer

def evaluate_stop_condition(model, tokenizer, input_ids, target_id, logit_importance_score):
    # Replace tokens with low importance score and then inference \hat{y^{(e)}_{t+1}}

    token_replacer = RankingTokenReplacer(UniformTokenSampler(tokenizer), 3)
    token_replacer.set_value(torch.softmax(logit_importance_score, 0))
    input_ids_replaced, mask_replacing = token_replacer.sample(input_ids)

    print(f"Replacing mask based on importance score: { mask_replacing }")

    # Whether the result \hat{y^{(e)}_{t+1}} consistent with y_{t+1}

    logits_replaced = model(input_ids_replaced)['logits'][0]
    prob_replaced_target = torch.softmax(logits_replaced[input_ids_replaced.shape[1] - 1], 0)[target_id]
    id_max_logits = torch.argmax(logits_replaced[-1])
    word_max_logits = tokenizer.decode([ id_max_logits ])
    
    print(f"Likelihood of the target: { prob_replaced_target }; Word with max likelihood: {word_max_logits}")

    # TODO: implement stop condition
    return False

def update_importance_score(logit_importance_score, model, tokenizer, input_ids, target_id, prob_original_target):
    # Randomly replace a set of tokens R to form a new sequence \hat{y_{1...t}}

    token_replacer = UniformTokenReplacer(UniformTokenSampler(tokenizer), 0.3)
    input_ids_replaced, mask_replacing = token_replacer.sample(input_ids)

    print(f"Replacing mask:     { [ mask_replacing[0, i].item() for i in range(mask_replacing.shape[1]) ] }")
    print(f"Replaced sequence:  { [ tokenizer.decode(input_ids_replaced[0, i]) for i in range(input_ids_replaced.shape[1]) ] }")
    
    # Inference \hat{p^{(y)}} = p(y_{t+1}|\hat{y_{1...t}})

    logits_replaced = model(input_ids_replaced)['logits'][0]
    prob_replaced_target = torch.softmax(logits_replaced[input_ids_replaced.shape[1] - 1], 0)[target_id]

    # Compute changes delta = p^{(y)} - \hat{p^{(y)}}

    delta_prob_target = prob_original_target - prob_replaced_target
    print(f"likelihood delta: { delta_prob_target }")

    # Update importance scores based on delta (magnitude) and replacement (direction)

    logit_importance_score = logit_importance_score.clone()
    logit_importance_score[mask_replacing[0] == 1] += delta_prob_target
    logit_importance_score[mask_replacing[0] == 0] -= delta_prob_target

    print(f"Updated importance score: { torch.softmax(logit_importance_score, 0) }")

    return logit_importance_score

@torch.no_grad()
def rationalize_random_replacing(model, tokenizer, input_ids, target_id):
    
    # Inference p^{(y)} = p(y_{t+1}|y_{1...t})

    logits_original = model(input_ids)['logits'][0]
    prob_original_target = torch.softmax(logits_original[input_ids.shape[1] - 1], 0)[target_id]

    # Initialize importance score s for each token in the sequence y_{1...t}

    logit_importance_score = torch.zeros([ input_ids.shape[1] ], device=input_ids.device)
    print(f"Initialize importance score:  { torch.softmax(logit_importance_score, 0) }")
    print()

    # TODO: limit max steps
    while True:
        
        # Update importance score
        logit_importance_score = update_importance_score(logit_importance_score, model, tokenizer, input_ids, target_id, prob_original_target)

        # Evaluate stop condition
        if evaluate_stop_condition(model, tokenizer, input_ids, target_id, logit_importance_score):
            break

    # TODO: return
    return None

if __name__ == '__main__':

    from transformers import AutoTokenizer, AutoModelWithLMHead

    # Load model from Hugging Face
    model = AutoModelWithLMHead.from_pretrained("gpt2-large")
    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")

    model.cuda()
    model.eval()

    input_string = "I love eating breakfast out the"
    t = tokenizer(input_string, return_tensors='pt')
    input_ids = tokenizer(input_string, return_tensors='pt')['input_ids'].to(model.device)
    generated_input = model.generate(input_ids=input_ids, max_length=8, do_sample=False)[0] 
    print(' generated input -->', tokenizer.decode(generated_input))

    # TODO: incomplete implementation
    rationalize_random_replacing(model, tokenizer, input_ids, generated_input[input_ids.shape[1]])
