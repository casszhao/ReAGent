import logging

import torch
from transformers import AutoModelWithLMHead, AutoTokenizer
from .base import BaseImportanceScoreEvaluator

import inseq

class InseqImportanceScoreEvaluator(BaseImportanceScoreEvaluator):
    """Importance Score Evaluator
    
    """

    def __init__(self, model: AutoModelWithLMHead, tokenizer: AutoTokenizer, method: str, attribute_params: dict) -> None:
        """Constructor

        Args:
            model: A Huggingface AutoModelWithLMHead model
            tokenizer: A Huggingface AutoTokenizer
            method: method

        """

        super().__init__(model, tokenizer)

        self.attribution_model = inseq.load_model(self.model.name_or_path, method)
        self.attribute_params = attribute_params

    def evaluate(self, input_ids: torch.Tensor, target_id: torch.Tensor) -> torch.Tensor:
        """Evaluate importance score of input sequence

        Args:
            input_ids: input sequence [batch, sequence]
            target_id: target token [batch]

        Return:
            importance_score: evaluated importance score for each token in the input [batch, sequence]

        """

        input_text = [ self.tokenizer.decode(i) for i in input_ids]
        target_text = [ self.tokenizer.decode(i) for i in torch.cat([input_ids, torch.unsqueeze(target_id, 0)], dim=1)]

        attr_res = self.attribution_model.attribute(
            input_text,
            target_text,
            **self.attribute_params
        )
        
        # [[ full_length, attr_length(1) ]]
        attrs_list = [ attr.aggregate().target_attributions[:-1] for attr in attr_res.sequence_attributions ]

        attrs_batch = torch.permute(torch.cat(attrs_list, dim=1), dims=[1, 0])

        self.important_score = attrs_batch
        return self.important_score
