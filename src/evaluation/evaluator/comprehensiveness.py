from typing_extensions import override
import torch
import torch.nn.functional as F
import torch.distributions as dist
from transformers import AutoModelForCausalLM
from .base_masking import BaseMaskingEvaluator

class ComprehensivenessEvaluator(BaseMaskingEvaluator):

    @override
    def __init__(self, model: AutoModelForCausalLM, rational_size: int = 0, rationale_ratio: float = 0) -> None:
        """ Constructor

        Args:
            model: AutoModelForCausalLM
            rational_size: number of rational tokens, rationale_ratio will be ignored
            rationale_ratio: ratio of rational tokens

        """
        super().__init__(model)
        self.rational_size = rational_size
        self.rationale_ratio = rationale_ratio

    @override
    def get_feature_masking_ratio(self, importance_scores: torch.Tensor) -> torch.Tensor:
        """ Get feature masking ratio for each token

        Args:
            importance_scores: importance_scores [batch, sequence]

        Return:
            feature_masking_ratio [batch, sequence]

        """
        top_k = self.rational_size
        if top_k <= 0:
            top_k = int(self.rationale_ratio * importance_scores.shape[1])

        binary_rational_mask = BaseMaskingEvaluator.create_binary_rational_mask(importance_scores, top_k)
        binary_nonrational_mask = 1 - binary_rational_mask
        return binary_nonrational_mask

    @override
    def get_metric(self, prob_original: torch.Tensor, prob_masked: torch.Tensor) -> torch.Tensor:
        """ Get metric score

        Args:
            prob_original: prob_original [batch, feature_out]
            prob_masked: prob_masked [batch, feature_out]

        Return:
            score [batch]

        """
        assert prob_masked.shape[0] == 1, "Invalid batch size > 1"
        assert prob_masked.shape[0] == 1, "Invalid batch size > 1"
        # TODO: why squeeze?

        # by cass not by batch --> squeezed
        p = prob_masked.squeeze()
        q = prob_original.squeeze()
        
        p = p / torch.sum(p)
        q = q / torch.sum(q)
        sqrt_p = torch.sqrt(p)
        sqrt_q = torch.sqrt(q)
        comprehensiveness = torch.norm(sqrt_p - sqrt_q) / torch.sqrt(torch.tensor(2.0))

        # normalized_cross_entropy  = torch.sum(q * (torch.log(q/p)))
        # comprehensiveness = max(0, normalized_cross_entropy)
        
        return comprehensiveness
