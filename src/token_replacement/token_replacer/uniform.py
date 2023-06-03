
import torch
from .base import TokenReplacer
from ..token_sampler.base import TokenSampler

class UniformTokenReplacer(TokenReplacer):
    def __init__(self, token_sampler: TokenSampler, ratio: float):
        super().__init__(token_sampler)
        self.ratio = ratio

    def sample(self, input):
        sample_uniform = torch.rand(input.shape, device=input.device)
        mask_replacing = sample_uniform < self.ratio

        token_sampled = self.token_sampler.sample(input)

        input_replaced = input * ~mask_replacing + token_sampled * mask_replacing

        return input_replaced