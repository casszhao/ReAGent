import torch
from transformers import AutoModelWithLMHead, AutoTokenizer
from typing_extensions import override

from .base import TokenSampler


class InferentialMTokenSampler(TokenSampler):
    """Sample tokens from a seq-2-seq model

    """

    @override
    def __init__(self, source_tokenizer: AutoTokenizer, sampler_tokenizer: AutoTokenizer, sampler_model: AutoModelWithLMHead) -> None:
        """Constructor

        Args:
            source_tokenizer: A Huggingface AutoTokenizer for decoding the inputs.
            sampler_tokenizer: A Huggingface AutoTokenizer for inference the output.
            sampler_model: A Huggingface AutoModelWithLMHead for inference the output.

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
            for pos_i in torch.arange(inputs.shape[1]):

                # first token
                if pos_i == 0:
                   seq_li.append(inputs[seq_i, 0])
                   continue

                # following tokens

                text_prefix = self.source_tokenizer.decode(inputs[seq_i, :pos_i])
                text_prefix_m = text_prefix.replace(self.source_tokenizer.eos_token, self.sampler_tokenizer.eos_token).replace(self.source_tokenizer.bos_token, self.sampler_tokenizer.bos_token)
                probe_prefix_m = torch.tensor([self.sampler_tokenizer.encode(text_prefix_m)], device=inputs.device)

                from transformers import RobertaTokenizerFast
                if isinstance(self.sampler_tokenizer, RobertaTokenizerFast):
                    probe_prefix_m = probe_prefix_m[:,:-1]  # trim EOS
                    
                output_replacing_m = self.sampler_model(probe_prefix_m)
                logits_replacing_m = output_replacing_m['logits']
                logits_replacing_m_last = logits_replacing_m[:,-1]
                id_infer_m = torch.argmax(logits_replacing_m_last, dim=-1)

                text_infer_m = self.sampler_tokenizer.decode(id_infer_m)
                text_infer = text_infer_m.replace(self.sampler_tokenizer.eos_token, self.source_tokenizer.eos_token).replace(self.sampler_tokenizer.bos_token, self.source_tokenizer.bos_token)
                id_infer = self.source_tokenizer.encode(text_infer)
                id_infer_fst = id_infer[0]

                seq_li.append(id_infer_fst)

            batch_li.append(seq_li)
        
        res = torch.tensor(batch_li, device=inputs.device)

        return res

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cpu"

    source_tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir="cache")
    source_model = AutoModelForCausalLM.from_pretrained("gpt2", cache_dir="cache").to(device)
    source_model.eval()

    sampler_tokenizer = AutoTokenizer.from_pretrained("roberta-base", cache_dir="cache")
    sampler_model = AutoModelForCausalLM.from_pretrained("roberta-base", cache_dir="cache").to(device)
    sampler_model.eval()

    sampler = InferentialMTokenSampler(source_tokenizer, sampler_tokenizer, sampler_model)

    text = "This is a test sequence"
    inputs = torch.tensor([ source_tokenizer.encode(text) ], device=device)

    outputs = sampler.sample(inputs)

    print(outputs)
    print(source_tokenizer.decode(outputs[0]))


