
import torch

class ImportanceScoreEvaluator():

    def __init__(self, model, tokenizer, token_replacer, stopping_condition_evaluator) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.token_replacer = token_replacer
        self.stopping_condition_evaluator = stopping_condition_evaluator
        pass

    def update_importance_score(self, logit_importance_score, input_ids, target_id, prob_original_target):
        # Randomly replace a set of tokens R to form a new sequence \hat{y_{1...t}}

        input_ids_replaced, mask_replacing = self.token_replacer.sample(input_ids)

        print(f"Replacing mask:     { [ mask_replacing[0, i].item() for i in range(mask_replacing.shape[1]) ] }")
        print(f"Replaced sequence:  { [ self.tokenizer.decode(input_ids_replaced[0, i]) for i in range(input_ids_replaced.shape[1]) ] }")
        
        # Inference \hat{p^{(y)}} = p(y_{t+1}|\hat{y_{1...t}})

        logits_replaced = self.model(input_ids_replaced)['logits']
        prob_replaced_target = torch.softmax(logits_replaced[:, input_ids_replaced.shape[1] - 1, :], -1)[:, target_id]

        # Compute changes delta = p^{(y)} - \hat{p^{(y)}}

        delta_prob_target = prob_original_target - prob_replaced_target
        print(f"likelihood delta: { delta_prob_target }")

        # Update importance scores based on delta (magnitude) and replacement (direction)

        delta_score = mask_replacing * delta_prob_target + ~mask_replacing * -delta_prob_target
        logit_importance_score = logit_importance_score + delta_score
        print(f"Updated importance score: { torch.softmax(logit_importance_score, -1) }")

        return torch.softmax(logit_importance_score, -1)

    def evaluate(self, input_ids, target_id):
        
        # Inference p^{(y)} = p(y_{t+1}|y_{1...t})

        logits_original = self.model(input_ids)['logits']
        prob_original_target = torch.softmax(logits_original[:, input_ids.shape[1] - 1, :], -1)[:, target_id]

        # Initialize importance score s for each token in the sequence y_{1...t}

        logit_importance_score = torch.zeros(input_ids.shape, device=input_ids.device)
        print(f"Initialize importance score:  { torch.softmax(logit_importance_score, -1) }")
        print()

        # TODO: limit max steps
        while True:
            
            # Update importance score
            logit_importance_score = self.update_importance_score(logit_importance_score, input_ids, target_id, prob_original_target)

            # Evaluate stop condition
            if self.stopping_condition_evaluator.evaluate(input_ids, target_id, torch.softmax(logit_importance_score, -1)):
                break

        return torch.softmax(logit_importance_score, -1)
