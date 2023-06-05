
import torch
from .base import TokenReplacer
from ..token_sampler.base import TokenSampler

class UniformTokenReplacer(TokenReplacer):
    """Replace tokens in a sequence where selecting is base on uniform distribution

    """
    def __init__(self, token_sampler: TokenSampler, ratio: float):
        """Constructor

        Args:
            token_sampler: A TokenSampler for sampling replace token.
            ratio: replace ratio
        """
        super().__init__(token_sampler)
        self.ratio = ratio

    def sample(self, input):
        """Sample a sequence

        Args:
            input: input sequence
        
        Returns:
            input_replaced: A replaced sequence
            mask_replacing: Identify which token has been replaced
        """
        sample_uniform = torch.rand(input.shape, device=input.device)
        mask_replacing = sample_uniform < self.ratio

        token_sampled = self.token_sampler.sample(input)

        input_replaced = input * ~mask_replacing + token_sampled * mask_replacing

        return input_replaced, mask_replacing
    