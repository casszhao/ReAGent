from typing_extensions import override
import torch
from transformers import AutoModelForCausalLM
from .base import BaseEvaluator
from .sufficiency import SufficiencyEvaluator
from .comprehensiveness import ComprehensivenessEvaluator

class NormalizedComprehensivenessEvaluator(BaseEvaluator):

    @override
    def __init__(self, model: AutoModelForCausalLM, rational_ratio: float) -> None:
        """ Constructor

        Args:
            model: AutoModelForCausalLM
            rational_ratio: ratio of rational tokens

        """
        super().__init__()
        self.model = model
        self.sufficiency_evaluator_0 = SufficiencyEvaluator(model, 0)
        self.comprehensiveness_evaluator = ComprehensivenessEvaluator(model, rational_ratio)

    
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
            #prob_target_original = prob_original[torch.arange(prob_original.shape[0]), target_id]
        

        comprehensiveness = self.comprehensiveness_evaluator.evaluate(input_ids, target_id, importance_scores, input_wte, prob_original)
        sufficiency_0 = self.sufficiency_evaluator_0.evaluate(input_ids, target_id, importance_scores, input_wte, prob_original)
        norm_comprehensiveness = comprehensiveness / (1 - sufficiency_0)
        
        return norm_comprehensiveness
