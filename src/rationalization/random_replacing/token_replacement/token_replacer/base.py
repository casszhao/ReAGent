from typing import Union
import torch
from ..token_sampler.base import TokenSampler
from utils.traceable import Traceable

class TokenReplacer(Traceable):
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
    
    def trace_start(self):
        """Start tracing
        
        """
        super().trace_start()
        self.token_sampler.trace_start()

    def trace_stop(self):
        """Stop tracing
        
        """
        super().trace_stop()
        self.token_sampler.trace_stop()
