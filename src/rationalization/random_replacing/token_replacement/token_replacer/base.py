import torch
from ..token_sampler.base import TokenSampler

class TokenReplacer():
    """
    Base class for token replacer
    """
    def __init__(self, token_sampler: TokenSampler) -> None:
        self.token_sampler = token_sampler

    def sample(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
