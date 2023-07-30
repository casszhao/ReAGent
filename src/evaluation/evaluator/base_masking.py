from typing_extensions import override
import torch
import torch.nn as nn
soft_max = nn.Softmax(dim=1)
from transformers import AutoModelForCausalLM, GPT2LMHeadModel, OPTForCausalLM
from .base import BaseEvaluator
import logging

class BaseMaskingEvaluator(BaseEvaluator):

    @override
    def __init__(self, model: AutoModelForCausalLM) -> None:
        """ Constructor

        Args:
            model: AutoModelForCausalLM

        """
        super().__init__()

        self.model = model

    def get_feature_masking_ratio(self, importance_scores: torch.Tensor) -> torch.Tensor:
        """ Get feature masking ratio for each token

        Args:
            importance_scores: importance_scores [batch, sequence]

        Return:
            feature_masking_ratio [batch, sequence]

        """
        raise NotImplementedError()

    def get_metric(self, prob_original: torch.Tensor, prob_masked: torch.Tensor) -> torch.Tensor:
        """ Get metric score

        Args:
            prob_original: prob_original [batch]
            prob_masked: prob_masked [batch]

        Return:
            score [batch]

        """
        raise NotImplementedError()
    
    @override
    @torch.no_grad()
    def evaluate(self, input_ids: torch.Tensor, target_id: torch.Tensor, importance_scores: torch.Tensor, input_wte: torch.Tensor = None, prob_original: torch.Tensor = None) -> torch.Tensor:
        """ Evaluate Comprehensiveness

        Args:
            input_ids: input token ids [batch, sequence]
            target_id: target token id [batch] (Deprecated)
            importance_scores: importance_scores of input tokens [batch, sequence]
            input_wte: input word token embedding (Optional)
            prob_original: probabilitise of original input (Optional)

        Return:
            score [batch]

        """
        

        if input_wte == None:

            if isinstance(self.model, GPT2LMHeadModel):
                gpt2Model: GPT2LMHeadModel = self.model
                input_wte = gpt2Model.transformer.wte.weight[input_ids,:]
            elif isinstance(self.model, OPTForCausalLM):
                optModel: OPTForCausalLM = self.model
                input_wte = optModel.model.decoder.embed_tokens(input_ids)
            else:
                raise ValueError(f"Unsupported model {type(self.model)}")

        # original prob
        if prob_original == None:
            logits_original = self.model(inputs_embeds=input_wte)["logits"]
            logging.debug(f"logits_original.shape ==>> {logits_original.shape}")
            prob_original = torch.softmax(logits_original[:, input_ids.shape[1] - 1, :], -1) 
            logging.debug(f"prob_original.shape ==>> {prob_original.shape}")
        
        # masked prob
        feature_masking_ratio = self.get_feature_masking_ratio(importance_scores)
        #logging.debug(f"==>> {torch.sum(feature_masking_ratio)}")  # testing if suff 0 masking all 0
        input_wte_masked = BaseMaskingEvaluator.mask_zero_embedding(input_wte, feature_masking_ratio)

        logits_masked = self.model(inputs_embeds=input_wte_masked)["logits"]
        prob_masked = torch.softmax(logits_masked[:, input_ids.shape[1] - 1, :], -1) 

        # metric
        metric = self.get_metric(prob_original, prob_masked)

        return metric

    def create_binary_rational_mask(importance_scores: torch.Tensor, top_k: float) -> torch.Tensor:
        """ Create a binary mask of rational tokens from importance score, where 1 for rational, 0 for non-rational
        
        Args: 
            importance_scores: importance_scores [batch, sequence]
            top_k: number of tokens to be masked

        Return:
            rational_mask: rational_mask [batch, sequence]

        """
        ranking = torch.argsort(importance_scores, dim=1, descending=True)
        mask_pos = ranking[:,:top_k]
        rational_mask = torch.zeros(importance_scores.shape, device=importance_scores.device).scatter(-1, mask_pos, 1)
        
        return rational_mask

    def mask_zero_embedding(embedding: torch.Tensor, token_mask_ratio: torch.Tensor) -> torch.Tensor:
        """
            Mask embedding elements to zeros with regards to a ratio for each token

        Args: 
            embedding: embedding to be masked [batch, sequence, feature]
            token_mask_ratio: masking ratio of each token [batch, sequence]

        Return:
            masked_embedding [batch, sequence, feature]

        """

        uniform_samples = torch.rand(embedding.shape, device=embedding.device)
        #uniform_samples = soft_max(uniform_samples)
        feature_mask = uniform_samples < torch.unsqueeze(token_mask_ratio, 2)
        masked_embedding = embedding * feature_mask

        return masked_embedding
