from typing_extensions import override
import torch
from transformers import AutoModelForCausalLM
from base_masking import BaseMaskingEvaluator

class SoftSufficiencyEvaluator(BaseMaskingEvaluator):    
    @override
    def __init__(self, model: AutoModelForCausalLM, rational_ratio: float) -> None:
        """ Constructor

        Args:
            model: AutoModelForCausalLM
            rational_ratio: ratio of rational tokens

        """
        super().__init__(model)
        self.rational_ratio = rational_ratio

    @override
    def get_feature_masking_ratio(self, importance_scores: torch.Tensor) -> torch.Tensor:
        """ Get feature masking ratio for each token

        Args:
            importance_scores: importance_scores [batch, sequence]

        Return:
            feature_masking_ratio [batch, sequence]

        """
        return 1 - importance_scores

    @override
    def get_metric(self, prob_target_original: torch.Tensor, prob_target_masked: torch.Tensor) -> torch.Tensor:
        """ Get metric score

        Args:
            prob_target_original: prob_target_original [batch]
            prob_target_masked: prob_target_masked [batch]

        Return:
            score [batch]

        """
        sufficiency = 1 - torch.max(torch.tensor(0, device=prob_target_original.device), prob_target_original - prob_target_masked)
        return sufficiency
