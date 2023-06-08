import torch

from token_replacement.token_sampler.uniform import UniformTokenSampler
from token_replacement.token_replacer.uniform import UniformTokenReplacer
from token_replacement.token_replacer.ranking import RankingTokenReplacer

class TopKStopConditionEvaluator():

    def __init__(self, model, token_sampler, top_k, tokenizer) -> None:
        
        self.model = model
        self.token_sampler = token_sampler
        self.top_k = top_k
        self.token_replacer = RankingTokenReplacer(self.token_sampler, self.top_k)
        self.tokenizer = tokenizer

    def evaluate(self, input_ids, target_id, logit_importance_score):
        # Replace tokens with low importance score and then inference \hat{y^{(e)}_{t+1}}
        
        self.token_replacer.set_score(torch.softmax(logit_importance_score, -1))
        input_ids_replaced, mask_replacing = self.token_replacer.sample(input_ids)

        print(f"Replacing mask based on importance score: { mask_replacing }")

        # Whether the result \hat{y^{(e)}_{t+1}} consistent with y_{t+1}

        logits_replaced = model(input_ids_replaced)['logits']

        ids_prediction_sorted = torch.argsort(logits_replaced[:,-1,:], descending=True)
        ids_prediction_top_k = ids_prediction_sorted[:, :self.top_k]

        top_k_words = [ [ self.tokenizer.decode([token_id]) for token_id in seq] for seq in ids_prediction_top_k ]
        print(top_k_words)

        match_mask = ids_prediction_top_k == target_id
        match_hit = torch.sum(match_mask, dim=-1)

        # TODO: batch incompatible - this only evaluate the first result in a batch
        return match_hit[0].item() > 0


class Rationalizer():

    def __init__(self) -> None:
        pass

    def update_importance_score(self, logit_importance_score, model, tokenizer, input_ids, target_id, prob_original_target):
        # Randomly replace a set of tokens R to form a new sequence \hat{y_{1...t}}

        token_replacer = UniformTokenReplacer(UniformTokenSampler(tokenizer), 0.5)
        input_ids_replaced, mask_replacing = token_replacer.sample(input_ids)

        print(f"Replacing mask:     { [ mask_replacing[0, i].item() for i in range(mask_replacing.shape[1]) ] }")
        print(f"Replaced sequence:  { [ tokenizer.decode(input_ids_replaced[0, i]) for i in range(input_ids_replaced.shape[1]) ] }")
        
        # Inference \hat{p^{(y)}} = p(y_{t+1}|\hat{y_{1...t}})

        logits_replaced = model(input_ids_replaced)['logits']
        prob_replaced_target = torch.softmax(logits_replaced[:, input_ids_replaced.shape[1] - 1, :], -1)[:, target_id]

        # Compute changes delta = p^{(y)} - \hat{p^{(y)}}

        delta_prob_target = prob_original_target - prob_replaced_target
        print(f"likelihood delta: { delta_prob_target }")

        # Update importance scores based on delta (magnitude) and replacement (direction)

        delta_score = mask_replacing * delta_prob_target + ~mask_replacing * -delta_prob_target
        logit_importance_score = logit_importance_score + delta_score
        print(f"Updated importance score: { torch.softmax(logit_importance_score, -1) }")

        return logit_importance_score

    @torch.no_grad()
    def rationalize_random_replacing(self, model, tokenizer, input_ids, target_id):
        
        # Inference p^{(y)} = p(y_{t+1}|y_{1...t})

        logits_original = model(input_ids)['logits']
        prob_original_target = torch.softmax(logits_original[:, input_ids.shape[1] - 1, :], -1)[:, target_id]

        # Initialize importance score s for each token in the sequence y_{1...t}

        logit_importance_score = torch.zeros(input_ids.shape, device=input_ids.device)
        print(f"Initialize importance score:  { torch.softmax(logit_importance_score, -1) }")
        print()

        # TODO: limit max steps
        while True:
            
            # Update importance score
            logit_importance_score = self.update_importance_score(logit_importance_score, model, tokenizer, input_ids, target_id, prob_original_target)

            # Evaluate stop condition
            stop_condition_evaluator = TopKStopConditionEvaluator(model, UniformTokenSampler(tokenizer), 3, tokenizer)
            if stop_condition_evaluator.evaluate(input_ids, target_id, logit_importance_score):
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

    input_string = [ 
        "I love eating breakfast out the"
    ]
    input_ids = tokenizer(input_string, return_tensors='pt')['input_ids'].to(model.device)
    generated_input = model.generate(input_ids=input_ids, max_length=8, do_sample=False) 
    print(' generated input -->', [ [ tokenizer.decode(token) for token in seq] for seq in generated_input ])

    target_id = generated_input[:, input_ids.shape[1]]
    print(' target -->', [ tokenizer.decode(token) for token in target_id ])

    # TODO: incomplete implementation
    # rationalize_random_replacing(model, tokenizer, input_ids, generated_input[input_ids.shape[1]])

    rationalizer = Rationalizer()
    rationalizer.rationalize_random_replacing(model, tokenizer, input_ids, generated_input[:, input_ids.shape[1]])
