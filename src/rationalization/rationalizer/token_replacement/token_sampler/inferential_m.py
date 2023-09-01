import torch
from transformers import AutoModelWithLMHead, AutoTokenizer, BertForMaskedLM
from typing_extensions import override

from .base import TokenSampler


class InferentialMTokenSampler(TokenSampler):
    """Sample tokens from a seq-2-seq model

    """

    @override
    def __init__(self, source_tokenizer: AutoTokenizer, sampler_tokenizer: AutoTokenizer, sampler_model: BertForMaskedLM) -> None:
        """Constructor

        Args:
            source_tokenizer: A Huggingface AutoTokenizer for decoding the inputs.
            sampler_tokenizer: A Huggingface AutoTokenizer for inference the output.
            sampler_model: A Huggingface BertForMaskedLM for inference the output.

        """
        super().__init__()

        self.source_tokenizer = source_tokenizer
        self.sampler_tokenizer = sampler_tokenizer
        self.sampler_model = sampler_model

    @override
    def sample(self, inputs: torch.Tensor) -> torch.Tensor:
        """Sample a tensor

        Args:
            inputs: input tensor [batch, sequence]
        
        Returns:
            token_inferences: sampled (placement) tokens by inference

        """
        super().sample(inputs)

        batch_li = []
        for seq_i in torch.arange(inputs.shape[0]):
            seq_li = []
            input_tokens = [ self.source_tokenizer.decode(i) for i in inputs[seq_i] ]
            for pos_i in torch.arange(inputs.shape[1]):
                tokens_masked = input_tokens[:]
                tokens_masked[pos_i] = self.sampler_tokenizer.mask_token

                text_masked = ''.join(tokens_masked)

                tokens_alt = self.sampler_tokenizer.tokenize(text_masked)
                masked_pos_alt = tokens_alt.index(self.sampler_tokenizer.mask_token)
                ids_alt = torch.tensor(self.sampler_tokenizer.convert_tokens_to_ids(tokens_alt), device=inputs.device)

                logits_pred_alt = self.sampler_model(torch.unsqueeze(ids_alt, 0))['logits'][0]
                logits_mask_pred_alt = logits_pred_alt[masked_pos_alt]

                id_pred_alt = torch.argmax(logits_mask_pred_alt, dim=-1)

                text_pred = self.sampler_tokenizer.convert_ids_to_tokens(id_pred_alt.item())
                id_pred = self.source_tokenizer.convert_tokens_to_ids(text_pred)
                id_pred_fst = id_pred

                seq_li.append(id_pred_fst)

            batch_li.append(seq_li)
        
        res = torch.tensor(batch_li, device=inputs.device)

        return res
