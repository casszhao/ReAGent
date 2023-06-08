import torch
from .base import StoppingConditionEvaluator
from token_replacement.token_sampler.base import TokenSampler
from token_replacement.token_replacer.ranking import RankingTokenReplacer
from transformers import AutoTokenizer, AutoModelWithLMHead

class TopKStoppingConditionEvaluator(StoppingConditionEvaluator):
    """
    Stopping Condition Evaluator which stop when target exist in top k predictions, 
    while top n tokens based on importance_score are not been replaced.
    """

    def __init__(self, model: AutoModelWithLMHead, token_sampler: TokenSampler, top_k: int, top_n: int = 0, top_n_ratio: float = 0, tokenizer: AutoTokenizer = None) -> None:
        self.model = model
        self.token_sampler = token_sampler
        self.top_k = top_k
        self.token_replacer = RankingTokenReplacer(self.token_sampler, top_n, top_n_ratio)
        self.tokenizer = tokenizer

    def evaluate(self, input_ids: torch.Tensor, target_id: torch.Tensor, importance_score: torch.Tensor) -> bool:
        # Replace tokens with low importance score and then inference \hat{y^{(e)}_{t+1}}
        
        self.token_replacer.set_score(importance_score)
        input_ids_replaced, mask_replacing = self.token_replacer.sample(input_ids)

        print(f"Replacing mask based on importance score: { mask_replacing }")

        # Whether the result \hat{y^{(e)}_{t+1}} consistent with y_{t+1}

        logits_replaced = self.model(input_ids_replaced)['logits']

        ids_prediction_sorted = torch.argsort(logits_replaced[:,-1,:], descending=True)
        ids_prediction_top_k = ids_prediction_sorted[:, :self.top_k]

        if self.tokenizer:
            top_k_words = [ [ self.tokenizer.decode([token_id]) for token_id in seq] for seq in ids_prediction_top_k ]
            print(top_k_words)

        match_mask = ids_prediction_top_k == target_id
        match_hit = torch.sum(match_mask, dim=-1)

        # Stops only when all the samples in a batch completed
        # TODO: optimization - stop/bypass part of the batch
        return torch.prod(match_hit) > 0
