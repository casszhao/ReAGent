import torch
from .base import TokenSampler
from transformers import AutoTokenizer, AutoModelWithLMHead

class InferentialTokenSampler(TokenSampler):
  """Sample tokens from a seq-2-seq model

  """
  def __init__(self, tokenizer: AutoTokenizer, model: AutoModelWithLMHead) -> None:
    """Constructor

    Args:
      tokenizer: A Huggingface AutoTokenizer.
      model: A Huggingface AutoModelWithLMHead for inference the output.

    """
    self.tokenizer = tokenizer
    self.model = model

  def sample(self, input: torch.Tensor) -> torch.Tensor:
    """Sample a tensor

    Args:
      input: input tensor [batch, sequence, feature]
    
    Returns:
      token_inferences: sampled (placement) tokens by inference
    """

    logits_replacing = self.model(input)['logits']
    ids_infer = torch.argmax(logits_replacing, dim=-1)

    token_inferences = torch.cat([ input[:, 0:1], ids_infer[:, :-1] ], dim=1)

    return token_inferences
