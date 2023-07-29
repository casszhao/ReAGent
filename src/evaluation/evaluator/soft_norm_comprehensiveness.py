from typing_extensions import override
import torch
from transformers import AutoModelForCausalLM
from .base import BaseEvaluator
from .sufficiency import SufficiencyEvaluator
from .comprehensiveness import ComprehensivenessEvaluator
from .soft_sufficiency import SoftSufficiencyEvaluator
from .soft_comprehensiveness import SoftComprehensivenessEvaluator

class SoftNormalizedComprehensivenessEvaluator(BaseEvaluator):

    @override
    def __init__(self, model: AutoModelForCausalLM) -> None:
        """ Constructor

        Args:
            model: AutoModelForCausalLM

        """
        super().__init__()
        self.model = model
        self.soft_sufficiency_evaluator_0 = SufficiencyEvaluator(model, rationale_ratio=0)
        #self.soft_sufficiency_evaluator_0 = SoftSufficiencyEvaluator(model)
        self.soft_comprehensiveness_evaluator = SoftComprehensivenessEvaluator(model)

    
    @torch.no_grad()
    def evaluate(self, input_ids: torch.Tensor, target_id: torch.Tensor, importance_scores: torch.Tensor, input_wte: torch.Tensor = None, prob_original: torch.Tensor = None) -> torch.Tensor:
        """ Evaluate Comprehensiveness

        Args:
            input_ids: input token ids [batch, sequence]
            target_id: target token id [batch] (Deprecated)
            importance_scores: importance_scores of input tokens [batch, sequence]
            input_wte: input word token embedding (Optional)
            prob_original: probabilities of original input (Optional)

        Return:
            score [batch]

        """

        if input_wte == None:
            input_wte = self.model.transformer.wte.weight[input_ids,:]

        # original prob
        if prob_original == None:
            logits_original = self.model(inputs_embeds=input_wte)["logits"]
            prob_original = torch.softmax(logits_original[:, input_ids.shape[1] - 1, :], -1)
        

        soft_comprehensiveness = self.soft_comprehensiveness_evaluator.evaluate(input_ids, None, importance_scores, input_wte, prob_original)
        print(' ')
        print(f"soft_comprehensiveness ==>> {soft_comprehensiveness}")
        soft_sufficiency_0 = self.soft_sufficiency_evaluator_0.evaluate(input_ids, None, importance_scores, input_wte, prob_original)
        print(f"soft_sufficiency_0 ==>> {soft_sufficiency_0}")
        soft_norm_comprehensiveness = soft_comprehensiveness / (1-soft_sufficiency_0)
        #soft_norm_comprehensiveness = soft_comprehensiveness
        return soft_norm_comprehensiveness
