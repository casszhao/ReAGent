
import math
from typing_extensions import override
import torch

from importance_score_evaluator import ImportanceScoreEvaluator
from utils.traceable import Traceable

class Rationalizer(Traceable):
    """Rationalizer
    
    """

    def __init__(self, importance_score_evaluator: ImportanceScoreEvaluator, top_n: float = 0, top_n_ratio: float = 0) -> None:
        """Constructor

        Args:
            importance_score_evaluator: A ImportanceScoreEvaluator
            top_n: Rational size
            top_n_ratio: Use ratio of sequence to define rational size

        """
        self.importance_score_evaluator = importance_score_evaluator
        self.top_n = top_n
        self.top_n_ratio = top_n_ratio

    def rationalize(self, input_ids: torch.Tensor, target_id: torch.Tensor) -> torch.Tensor:
        """Compute rational of a sequence on a target

        Args:
            input_ids: The sequence [batch, sequence]
            target_id: The target [batch]

        Return:
            pos_top_n: rational position in the sequence [batch, rational_size]

        """
        importance_score = self.importance_score_evaluator.evaluate(input_ids, target_id)
        
        pos_sorted = torch.argsort(importance_score, dim=-1, descending=True)

        top_n = self.top_n

        if top_n == 0:
            top_n = int(math.ceil(self.top_n_ratio * input_ids.shape[-1]))
            
        pos_top_n = pos_sorted[:, :top_n]

        return pos_top_n

    @override
    def trace_start(self) -> None:
        """Start tracing
        
        """
        super().trace_start()

        self.importance_score_evaluator.trace_start()

    @override
    def trace_stop(self) -> None:
        """Stop tracing
        
        """
        super().trace_stop()

        self.importance_score_evaluator.trace_stop()
