import torch

class TokenSampler():
  """Base class for token samplers

  """

  def __init__(self) -> None:
    """Constructor
    
    """
    raise NotImplementedError("This is the base class for token sampler, please use an actual class (e.g. UniformTokenSampler)")

  def sample(self, input: torch.Tensor) -> torch.Tensor:
    """Dummy sample
    
    """
    raise NotImplementedError()
