from typing_extensions import override
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base import BaseEvaluator
from .sufficiency import Suff_Evaluator
from .comprehensiveness import Comp_Evaluator

class Norm_Suff_Evaluator(BaseEvaluator):

    @override
    def __init__(self, model: AutoModelForCausalLM, rational_ratio: float) -> None:
        """ Constructor

        Args:
            model: AutoModelForCausalLM
            rational_ratio: ratio of rational tokens

        """
        super().__init__()
        self.model = model
        self.sufficiency_evaluator_0 = Suff_Evaluator(model, 0)
        self.sufficiency_evaluator = Suff_Evaluator(model, rational_ratio)

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
            prob_target_original = self.model.generate(inputs_embeds=input_wte)["logits"]  # by cass
            #prob_target_original = self.model.generate(input_ids, max_new_tokens=1)["logits"] 
            print(' ')
            print(' ')
            print(f"==>> prob_target_original: {prob_target_original}")
            quit()
            # prob_original = torch.softmax(logits_original[:, input_ids.shape[1] - 1, :], -1)  # commentout by cass
            # prob_target_original = prob_original[torch.arange(prob_original.shape[0]), target_id] # commentout by cass
        
        sufficiency = self.sufficiency_evaluator.evaluate(input_ids, target_id, importance_scores, input_wte, prob_target_original)
        print(f"==>> sufficiency for NORM: {sufficiency}")

        sufficiency_0 = self.sufficiency_evaluator_0.evaluate(input_ids, target_id, importance_scores, input_wte, prob_target_original)
        print(f"==>> sufficiency_0  for NORM: {sufficiency_0}")
        norm_sufficiency = (sufficiency - sufficiency_0) / (1 - sufficiency_0)
        print(f"==>> norm_sufficiency: {norm_sufficiency}")
        
        return norm_sufficiency

if __name__ == "__main__":

    from transformers import AutoModelWithLMHead, AutoTokenizer
    
    model = AutoModelWithLMHead.from_pretrained("gpt2-medium")
    tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")

    model.cuda()
    model.eval()
    
    # ======== prepare data ========

    # batch with size 1
    input_string = [
        # "I love eating breakfast in the",
        "When my flight landed in Thailand. I was staying in the capital city of"
        # "When my flight landed in Thailand, I converted my currency and slowly fell asleep. I was staying in the capital city of"
        # "When my flight landed in Thailand, I converted my currency and slowly fell asleep. (I had a terrifying dream about my grandmother, but that's a story for another time). I was staying in the capital city of"
    ]

    # generate prediction 
    input_ids = tokenizer(input_string, return_tensors='pt')['input_ids'].to(model.device)
    generated_input = model.generate(input_ids=input_ids, max_length=80, do_sample=False) 

    # extract target from prediction
    target_id = generated_input[:, input_ids.shape[1]]

    importance_scores = torch.softmax(torch.tensor([
        [ 0, 0, 0, 0, 500, 1000, 0, 0, 0, 0, -500, -500, 1000, 1000, 1000, ],
        [ 0, 0, 0, 0, -500, -1000, 0, 0, 0, 0, -500, -500, -1000, -1000, -1000, ],
    ], dtype=torch.float, device=input_ids.device), -1)


    evaluator = Norm_Suff_Evaluator(model, 0.5)
    metric = evaluator.evaluate(input_ids, target_id, importance_scores)

