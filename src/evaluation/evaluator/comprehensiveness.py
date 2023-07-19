from typing_extensions import override
import torch
from transformers import AutoModelForCausalLM
from .base_masking import BaseMaskingEvaluator

class ComprehensivenessEvaluator(BaseMaskingEvaluator):

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
        top_k = int(self.rational_ratio * importance_scores.shape[1])
        binary_rational_mask = BaseMaskingEvaluator.create_binary_rational_mask(importance_scores, top_k)
        binary_nonrational_mask = 1 - binary_rational_mask
        return binary_nonrational_mask

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
        comprehensiveness = max(0, normalized_cross_entropy)
        
        return comprehensiveness
