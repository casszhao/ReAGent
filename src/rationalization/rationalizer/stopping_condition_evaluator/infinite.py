import torch
from typing_extensions import override

from .base import StoppingConditionEvaluator


class InfiniteStoppingConditionEvaluator(StoppingConditionEvaluator):
    """
    Stopping Condition Evaluator which never stop.
    """

    @override
    def __init__(self) -> None:
        """Constructor

        """
        super().__init__()

    @override
    def evaluate(self, input_ids: torch.Tensor, target_id: torch.Tensor, importance_score: torch.Tensor) -> torch.Tensor:
        """Evaluate stop condition

        Args:
            input_ids: Input sequence [batch, sequence]
            target_id: Target token [batch]
            importance_score: Importance score of the input [batch, sequence]

        Return:
            Whether the stop condition achieved [batch]

        """
        super().evaluate(input_ids, target_id, importance_score)

        match_hit = torch.zeros([input_ids.shape[0]], dtype=torch.bool, device=input_ids.device)

        # Stop flags for each sample in the batch
        return match_hit
