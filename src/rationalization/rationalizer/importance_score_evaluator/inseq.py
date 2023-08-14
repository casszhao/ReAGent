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
        ## for gpt j ===> \n token id is 198  \t 197  , space 220, \n\n token id is 
        input_text = [ self.tokenizer.decode(i) for i in input_ids]
        target_text = [ self.tokenizer.decode(i) for i in torch.cat([input_ids, torch.unsqueeze(target_id, 0)], dim=1)]


        # print( )
        # print( )
        # print(f"input_text: {input_text}")
        
        # print( )
        # print(f"target_text: {target_text}")
        # print(' target_id  ===>', target_id)

        # if '\n' in target_text[0][:-2]: # target_id[0].item() == 198:
        #     print(' contain !!!! ', target_text[0][:-2])
        #     input_text[0] = input_text[0][:-1]
        #     target_text[0] = target_text[0][:-1]

        

        # debugging from below for \n \n issue by cass
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
