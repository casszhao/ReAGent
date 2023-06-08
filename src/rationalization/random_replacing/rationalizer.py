
import math
import torch

class Rationalizer():

    def __init__(self, importance_score_evaluator, top_n: float = 0, top_n_ratio: float = 0) -> None:
        self.importance_score_evaluator = importance_score_evaluator
        self.top_n = top_n
        self.top_n_ratio = top_n_ratio

    def rationalize(self, input_ids, target_id):
        importance_score = self.importance_score_evaluator.evaluate(input_ids, target_id)
        
        pos_sorted = torch.argsort(importance_score, dim=-1, descending=True)

        top_n = self.top_n

        if top_n == 0:
            top_n = int(math.ceil(self.top_n_ratio * input_ids.shape[-1]))
            
        pos_top_n = pos_sorted[:, :top_n]

        return pos_top_n
