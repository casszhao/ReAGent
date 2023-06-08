from typing import Union
import torch
from ..token_sampler.base import TokenSampler

class TokenReplacer():
    """
    Base class for token replacers

    """

    def __init__(self, token_sampler: TokenSampler) -> None:
        """Constructor
        
        """
        self.token_sampler = token_sampler

    def sample(self, input: torch.Tensor) -> Union[torch.Tensor, torch.Tensor]:
        """Dummy sample

        """
        raise NotImplementedError()
