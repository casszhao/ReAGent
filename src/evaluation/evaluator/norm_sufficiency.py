from typing_extensions import override
import torch
from transformers import AutoModelForCausalLM, GPT2LMHeadModel, OPTForCausalLM
from .base import BaseEvaluator
from .sufficiency import SufficiencyEvaluator
from .comprehensiveness import ComprehensivenessEvaluator
import logging

class NormalizedSufficiencyEvaluator(BaseEvaluator):

    @override
    def __init__(self, model: AutoModelForCausalLM, rational_size: int = 0, rationale_ratio: float = 0) -> None:
        """ Constructor

        Args:
            model: AutoModelForCausalLM
            rational_size: number of rational tokens, rationale_ratio will be ignored
            rationale_ratio: ratio of rational tokens

        """
        super().__init__()
        self.model = model
        self.sufficiency_evaluator_0 = SufficiencyEvaluator(model, rationale_ratio=0)
        self.sufficiency_evaluator = SufficiencyEvaluator(model, rational_size, rationale_ratio)

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
            prob_original = torch.softmax(logits_original[:, input_ids.shape[1] - 1, :], -1)
        

        sufficiency = self.sufficiency_evaluator.evaluate(input_ids, None, importance_scores, input_wte, prob_original)
        logging.debug(' ')
        logging.debug(f"sufficiency==>> {sufficiency}")
        sufficiency_0 = self.sufficiency_evaluator_0.evaluate(input_ids, None, importance_scores, input_wte, prob_original)
        logging.debug(f"sufficiency_0==>> {sufficiency_0}")
        #norm_sufficiency = torch.clamp((sufficiency - sufficiency_0), min=0, max=10) / (1 - sufficiency_0)
        #norm_sufficiency = sufficiency
        norm_sufficiency = max(0,sufficiency-sufficiency_0)/(1-sufficiency_0)
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
    print(' generated input -->', [ [ tokenizer.decode(token) for token in seq] for seq in generated_input ])

    # extract target from prediction
    target_id = generated_input[:, input_ids.shape[1]]
    print(' target -->', [ tokenizer.decode(token) for token in target_id ])

    importance_scores = torch.softmax(torch.tensor([
        [ 0, 0, 0, 0, 500, 1000, 0, 0, 0, 0, -500, -500, 1000, 1000, 1000, ],
        [ 0, 0, 0, 0, -500, -1000, 0, 0, 0, 0, -500, -500, -1000, -1000, -1000, ],
    ], dtype=torch.float, device=input_ids.device), -1)


    evaluator = NormalizedSufficiencyEvaluator(model, 0.5)
    metric = evaluator.evaluate(input_ids, target_id, importance_scores)

    print(metric)