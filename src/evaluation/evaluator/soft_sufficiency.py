from typing_extensions import override
import torch
from transformers import AutoModelForCausalLM
from .base_masking import BaseMaskingEvaluator

class SoftSufficiencyEvaluator(BaseMaskingEvaluator):    
    @override
    def __init__(self, model: AutoModelForCausalLM) -> None:
        """ Constructor

        Args:
            model: AutoModelForCausalLM

        """
        super().__init__(model)

    @override
    def get_feature_masking_ratio(self, importance_scores: torch.Tensor) -> torch.Tensor:
        """ Get feature masking ratio for each token

        Args:
            importance_scores: importance_scores [batch, sequence]

        Return:
            feature_masking_ratio [batch, sequence]

        """
        return importance_scores

    @override
    def get_metric(self, prob_target_original: torch.Tensor, prob_target_masked: torch.Tensor) -> torch.Tensor:
        """ Get metric score

        Args:
            prob_target_original: prob_target_original [batch]
            prob_target_masked: prob_target_masked [batch]

        Return:
            score [batch]

        """
        
        # by cass not by batch --> squeezed
        p = prob_target_masked.squeeze()
        q = prob_target_original.squeeze()
        entropy = torch.nn.functional.kl_div(torch.log(q), p, reduction='sum')
        normalized_cross_entropy = entropy / torch.log(torch.tensor(q.size()[0], dtype=torch.float32)) # to normalise to make sure the range of entropy between 0 -1
        sufficiency = 1 - max(0, normalized_cross_entropy)
        print(f"==>> soft sufficiency: {sufficiency}")

        return sufficiency
