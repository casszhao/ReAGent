from typing import Any

import torch

from fairseq.models.transformer import TransformerModel

from tokenizer import FairseqTokenizerWrapper


class FairseqModelWrapper:

    def __init__(self, fairseq_model: TransformerModel) -> None:
        self.fairseq_model = fairseq_model
    
    def __call__(self, encoder_ids: torch.Tensor, decoder_ids: torch.Tensor) -> torch.Tensor:
        encoder_out = self.fairseq_model.models[0].encoder(encoder_ids)
        decoder_out = self.fairseq_model.models[0].decoder(decoder_ids, encoder_out=encoder_out)
        return {
            'logits': decoder_out[0]
        }

if __name__ == "__main__":

    fairseq_model = TransformerModel.from_pretrained(
        "../sequential-rationales/fairseq/.../compatible_iwslt",
        checkpoint_file='checkpoint_best.pt',
        data_name_or_path='./../../data-bin/iwslt14_distractors.tokenized.de-en')
    
    model = FairseqModelWrapper(fairseq_model)
    tokenizer = FairseqTokenizerWrapper(fairseq_model)

    source_text = "<s> es ist interessant . </s>"
    # target_text_full = "<s> it &apos;s interesting . </s>"
    target_text = "<s> it &apos;s"

    source_ids = tokenizer.encode(source_text, source=True)
    target_ids = tokenizer.encode(target_text)

    source_batch = torch.tensor([source_ids])
    target_batch = torch.tensor([target_ids])

    output_target_probs = model(source_batch, target_batch)["logits"]
    output_target_ids = torch.argmax(output_target_probs, dim=-1)
    output_target_text = tokenizer.decode(output_target_ids[0])
    
    print()
    print("================")
    print()
    print(output_target_text)
    print()
