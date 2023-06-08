import torch

class TokenSampler():
  """Base class for token sampler

  """

  def __init__(self) -> None:
    raise NotImplementedError("This is the base class for token sampler, please use an actual class (e.g. UniformTokenSampler)")

  def sample(self, input: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError()
