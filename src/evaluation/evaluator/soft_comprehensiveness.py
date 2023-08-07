from typing_extensions import override
import torch
import torch.nn.functional as F
import torch.distributions as dist
from transformers import AutoModelForCausalLM
from .base_masking import BaseMaskingEvaluator

class SoftComprehensivenessEvaluator(BaseMaskingEvaluator):

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
        return 1 - importance_scores

    @override
    def get_metric(self, prob_original: torch.Tensor, prob_masked: torch.Tensor) -> torch.Tensor:
        """ Get metric score

        Args:
            prob_original: prob_original [batch]
            prob_masked: prob_masked [batch]

        Return:
            score [batch]

        """
        # by cass not by batch --> squeezed
        p = prob_masked.squeeze()
        q = prob_original.squeeze()
        
        # p = p / torch.sum(p)
        # q = q / torch.sum(q)
        # sqrt_p = torch.sqrt(p)
        # sqrt_q = torch.sqrt(q)
        # comprehensiveness = torch.norm(sqrt_p - sqrt_q) / torch.sqrt(torch.tensor(2.0))

        sqrt_p = torch.sqrt(p)
        sqrt_q = torch.sqrt(q)
        distance = torch.sum( torch.pow((sqrt_p - sqrt_q), 2)  / torch.sqrt(torch.tensor(2.0)) )

        comprehensiveness = distance
        

        #normalized_cross_entropy  = torch.sum(q * (torch.log(q/p)))
        #comprehensiveness = max(0, normalized_cross_entropy)

        return comprehensiveness
