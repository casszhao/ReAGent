from typing_extensions import override
import torch
import torch.nn.functional as F
import torch.distributions as dist
from transformers import AutoModelForCausalLM
from .base_masking import BaseMaskingEvaluator

class SufficiencyEvaluator(BaseMaskingEvaluator):    
    @override
    def __init__(self, model: AutoModelForCausalLM, rational_size: int = 0, rational_ratio: float = 0) -> None:
        """ Constructor

        Args:
            model: AutoModelForCausalLM
            rational_size: number of rational tokens, rational_ratio will be ignored
            rational_ratio: ratio of rational tokens

        """
        super().__init__(model)
        self.rational_size = rational_size
        self.rational_ratio = rational_ratio

    @override
    def get_feature_masking_ratio(self, importance_scores: torch.Tensor) -> torch.Tensor:
        """ Get feature masking ratio for each token

        Args:
            importance_scores: importance_scores [batch, sequence]

        Return:
            feature_masking_ratio [batch, sequence]

        """
        top_k = self.rational_size
        if top_k <= 0:
            top_k = int(self.rational_ratio * importance_scores.shape[1])

        binary_rational_mask = BaseMaskingEvaluator.create_binary_rational_mask(importance_scores, top_k)
        return binary_rational_mask

    @override
    def get_metric(self, prob_original: torch.Tensor, prob_masked: torch.Tensor) -> torch.Tensor:
        """ Get metric score

        Args:
            prob_original: prob_original [batch]
            prob_masked: prob_masked [batch]

        Return:
            score [batch]

        """

        # by cass not by batch --> squeezed
        p = prob_masked.squeeze()
        q = prob_original.squeeze()
        # entropy = torch.nn.functional.kl_div(torch.log(q), p, reduction='sum')
        # normalized_cross_entropy = entropy / torch.log(torch.tensor(q.size()[0], dtype=torch.float32)) # to normalise to make sure the range of entropy between 0 -1

        normalized_cross_entropy  = torch.sum(q * (torch.log(q/p)))
        sufficiency = 1 - max(0, normalized_cross_entropy)

        return sufficiency







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
    generated_input = model.generate(input_ids=input_ids, do_sample=False) 
    generated_logits = model(input_ids=input_ids)['logits']
    print(f"==>> generated_logits: {generated_logits}")
    #predicted_token_logits = 
    print(' generated input -->', [ [ tokenizer.decode(token) for token in seq] for seq in generated_input ])

    # extract target from prediction
    target_id = generated_input[:, input_ids.shape[1]]
    print(' target -->', [ tokenizer.decode(token) for token in target_id ])

    importance_scores = torch.softmax(torch.tensor([
        [ 0, 0, 0, 0, 500, 1000, 0, 0, 0, 0, -500, -500, 1000, 1000, 1000, ],
        [ 0, 0, 0, 0, -500, -1000, 0, 0, 0, 0, -500, -500, -1000, -1000, -1000, ],
    ], dtype=torch.float, device=input_ids.device), -1)


    evaluator = SufficiencyEvaluator(model, 0.9)
    metric = evaluator.evaluate(input_ids, target_id, importance_scores)

    print(metric)
