from typing_extensions import override
import torch
from transformers import AutoModelForCausalLM
from .base import BaseEvaluator
from .sufficiency import Suff_Evaluator
from .comprehensiveness import Comp_Evaluator
from .soft_sufficiency import Soft_Suff_Evaluator
from .soft_comprehensiveness import Soft_Comp_Evaluator

class Norm_Soft_Suff_Evaluator(BaseEvaluator):

    @override
    def __init__(self, model: AutoModelForCausalLM) -> None:
        """ Constructor

        Args:
            model: AutoModelForCausalLM
            rational_ratio: ratio of rational tokens

        """
        super().__init__()
        self.model = model
        self.sufficiency_evaluator_0 = Suff_Evaluator(model, 0)
        self.soft_sufficiency_evaluator = Soft_Suff_Evaluator(model)

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
            prob_target_original = self.model.generate(inputs_embeds=input_wte)["logits"]
            # prob_original = torch.softmax(logits_original[:, input_ids.shape[1] - 1, :], -1)
            # prob_target_original = prob_original[torch.arange(prob_original.shape[0]), target_id]
        

        soft_sufficiency = self.soft_sufficiency_evaluator.evaluate(input_ids, target_id, importance_scores, input_wte, prob_target_original)
        sufficiency_0 = self.sufficiency_evaluator_0.evaluate(input_ids, target_id, importance_scores, input_wte, prob_target_original)
        soft_norm_sufficiency = (soft_sufficiency - sufficiency_0) / (1 - sufficiency_0)
        
        return soft_norm_sufficiency
