
import torch
from .base import TokenReplacer
from ..token_sampler.base import TokenSampler

class ThresholdTokenReplacer(TokenReplacer):
    """Replace tokens in a sequence based on a threshold

    """
    def __init__(self, token_sampler: TokenSampler, threshold: float, replace_greater: bool = False):
        """Constructor

        Args:
            token_sampler: A TokenSampler for sampling replace token.
            threshold: replacing threshold, 
        """
        super().__init__(token_sampler)
        self.threshold = threshold
        self.replace_greater = replace_greater

    def set_value(self, value):
        if not self.replace_greater:
            self.mask_replacing = value < self.threshold
        else:
            self.mask_replacing = value > self.threshold

    def sample(self, input):
        """Sample a sequence

        Args:
            input: input sequence
        
        Returns:
            input_replaced: A replaced sequence
            mask_replacing: Identify which token has been replaced
        """

        token_sampled = self.token_sampler.sample(input)

        input_replaced = input * ~self.mask_replacing + token_sampled * self.mask_replacing

        return input_replaced, self.mask_replacing

    