import torch


class StoppingConditionEvaluator():
    def __init__(self):
        raise NotImplementedError()

    def evaluate(self, input_ids: torch.Tensor, target_id: torch.Tensor, importance_score: torch.Tensor) -> bool:
        raise NotImplementedError()
