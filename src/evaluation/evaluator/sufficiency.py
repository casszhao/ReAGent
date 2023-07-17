from typing_extensions import override
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from .base_masking import BaseMaskingEvaluator

class Suff_Evaluator(BaseMaskingEvaluator):   # class BaseMaskingEvaluator(BaseEvaluator)
    @override
    def __init__(self, model: AutoModelForCausalLM, rational_ratio: float) -> None:
        """ Constructor

        Args:
            model: AutoModelForCausalLM
            rational_ratio: ratio of rational tokens

        """
        super().__init__(model)
        self.rational_ratio = rational_ratio

    @override
    def get_feature_masking_ratio(self, importance_scores: torch.Tensor) -> torch.Tensor:
        """ Get feature masking ratio for each token

        Args:
            importance_scores: importance_scores [batch, sequence]

        Return:
            feature_masking_ratio [batch, sequence]

        """
        top_k = int(self.rational_ratio * importance_scores.shape[1])
        binary_rational_mask = BaseMaskingEvaluator.create_binary_rational_mask(importance_scores, top_k)
        return binary_rational_mask

    @override
    def get_metric(self, prob_target_original: torch.Tensor, prob_target_masked: torch.Tensor) -> torch.Tensor:
        """ Get metric score

        Args:
            prob_target_original: prob_target_original [batch]
            prob_target_masked: prob_target_masked [batch]

        Return:
            score [batch]

        by cass ===> 
        we calcualte the cross entropy between two distribution

        for SUFF, if it is faithful, we expect a similar distribution, therefore, small cross entropy, "low loss". 
        so, similar to the original suff, the lower the better before normalised. 
        <===== 

        """
        #sufficiency = 1 - torch.max(torch.tensor(0, device=prob_target_original.device), prob_target_original - prob_target_masked)

        # q = prob_target_original
        # p = prob_target_masked

        print('prob_target_original ', prob_target_original)
        print('prob_target_original ', prob_target_original[0])
        print('prob_target_masked ', prob_target_masked)

        sufficiency = 1 - torch.max(torch.tensor(0, device=prob_target_original.device), 
        
                                    torch.tensor(F.kl_div(prob_target_original[0].log(), prob_target_masked[0], reduction='sum'), device=prob_target_original.device))
        print(f"==>> sufficiency <===: {sufficiency}")
        

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

    # extract target from prediction
    target_id = generated_input[:, input_ids.shape[1]]

    importance_scores = torch.softmax(torch.tensor([
        [ 0, 0, 0, 0, 500, 1000, 0, 0, 0, 0, -500, -500, 1000, 1000, 1000, ],
        [ 0, 0, 0, 0, -500, -1000, 0, 0, 0, 0, -500, -500, -1000, -1000, -1000, ],
    ], dtype=torch.float, device=input_ids.device), -1)


    evaluator = Suff_Evaluator(model, 0.9)
    metric = evaluator.evaluate(input_ids, target_id, importance_scores)

