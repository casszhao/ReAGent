from typing_extensions import override
import torch
from transformers import AutoModelForCausalLM
from base import BaseEvaluator

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

    def get_metric(self, prob_target_original: torch.Tensor, prob_target_masked: torch.Tensor) -> torch.Tensor:
        """ Get metric score

        Args:
            prob_target_original: prob_target_original [batch]
            prob_target_masked: prob_target_masked [batch]

        Return:
            score [batch]

        """
        raise NotImplementedError()
    
    @override
    @torch.no_grad()
    def evaluate(self, input_ids: torch.Tensor, target_id: torch.Tensor, importance_scores: torch.Tensor, input_wte: torch.Tensor = None, prob_target_original: torch.Tensor = None) -> torch.Tensor:
        """ Evaluate Comprehensiveness

        Args:
            input_ids: input token ids [batch, sequence]
            target_id: target token id [batch]
            importance_scores: importance_scores of input tokens [batch, sequence]
            input_wte: input word token embedding (Optional)
            prob_target_original: probability of target token of original input (Optional)

        Return:
            score [batch]

        """
        
        if input_wte == None:
            input_wte = self.model.transformer.wte.weight[input_ids,:]

        # original prob
        if prob_target_original == None:
            logits_original = self.model(inputs_embeds=input_wte)["logits"]
            prob_original = torch.softmax(logits_original[:, input_ids.shape[1] - 1, :], -1)
            prob_target_original = prob_original[torch.arange(prob_original.shape[0]), target_id]
        
        # masked prob
        feature_masking_ratio = self.get_feature_masking_ratio(importance_scores)
        input_wte_masked = BaseMaskingEvaluator.zero_mask_embedding(input_wte, feature_masking_ratio)

        logits_masked = self.model(inputs_embeds=input_wte_masked)["logits"]
        prob_masked = torch.softmax(logits_masked[:, input_ids.shape[1] - 1, :], -1)
        prob_target_masked = prob_masked[torch.arange(prob_masked.shape[0]), target_id]

        # metric
        metric = self.get_metric(prob_target_original, prob_target_masked)

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
        rational_mask = torch.ones(importance_scores.shape, device=importance_scores.device).scatter(-1, mask_pos, 0)
        
        return rational_mask

    def zero_mask_embedding(embedding: torch.Tensor, token_mask_ratio: torch.Tensor) -> torch.Tensor:
        """
            Mask embedding elements to zeros with regards to a ratio for each token

        Args: 
            embedding: embedding to be masked [batch, sequence, feature]
            token_mask_ratio: masking ratio of each token [batch, sequence]

        Return:
            masked_embedding [batch, sequence, feature]

        """

        feature_mask = torch.rand(embedding.shape, device=embedding.device) < torch.unsqueeze(token_mask_ratio, 2)
        masked_embedding = embedding * feature_mask

        return masked_embedding
