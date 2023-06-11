
import math
from typing_extensions import override
import torch

from importance_score_evaluator import ImportanceScoreEvaluator
from utils.traceable import Traceable

class AggregateRationalizer(Traceable):
    """AggregateRationalizer
    
    """

    def __init__(self, importance_score_evaluator: ImportanceScoreEvaluator, batch_size: int, occurrence_threshold: int, top_n: float = 0, top_n_ratio: float = 0) -> None:
        """Constructor

        Args:
            importance_score_evaluator: A ImportanceScoreEvaluator
            top_n: Rational size
            top_n_ratio: Use ratio of sequence to define rational size

        """
        self.importance_score_evaluator = importance_score_evaluator
        self.batch_size = batch_size
        self.occurrence_threshold = occurrence_threshold
        self.top_n = top_n
        self.top_n_ratio = top_n_ratio

    def rationalize(self, input_ids: torch.Tensor, target_id: torch.Tensor) -> torch.Tensor:
        """Compute rational of a sequence on a target

        Args:
            input_ids: The sequence [batch, sequence] (first dimension need to be 1)
            target_id: The target [batch]

        Return:
            pos_top_n: rational position in the sequence [batch, rational_size]

        """
        batch_input_ids = input_ids.repeat(self.batch_size, 1)

        importance_score = self.importance_score_evaluator.evaluate(batch_input_ids, target_id)
        self.importance_score_mean = torch.mean(importance_score, dim=0)
        
        pos_sorted = torch.argsort(importance_score, dim=-1, descending=True)

        top_n = self.top_n

        if top_n == 0:
            top_n = int(math.ceil(self.top_n_ratio * input_ids.shape[-1]))
            
        pos_top_n = pos_sorted[:, :top_n]

        count_occurrence = torch.bincount(pos_top_n.flatten(), minlength=input_ids.shape[1])
        pos_top_n = torch.unsqueeze(torch.nonzero(count_occurrence >= self.occurrence_threshold, as_tuple=True)[0], 0)

        return pos_top_n

    @override
    def trace_start(self) -> None:
        """Start tracing
        
        """
        super().trace_start()

        self.importance_score_evaluator.trace_start()

    @override
    def trace_stop(self) -> None:
        """Stop tracing
        
        """
        super().trace_stop()

        self.importance_score_evaluator.trace_stop()


@torch.no_grad()
def main():

    from transformers import AutoTokenizer, AutoModelWithLMHead
    from importance_score_evaluator import ImportanceScoreEvaluator
    from token_replacement.token_sampler.postag import POSTagTokenSampler
    from utils.serializing import serialize_rational
    from token_replacement.token_sampler.inferential import InferentialTokenSampler
    from stopping_condition_evaluator.top_k import TopKStoppingConditionEvaluator
    from token_replacement.token_sampler.uniform import UniformTokenSampler
    from token_replacement.token_replacer.uniform import UniformTokenReplacer

    # ======== model loading ========

    # Load model from Hugging Face
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

    # ======== hyper-parameters ========

    # replacing ratio during importance score updating
    updating_replacing_ratio = 0.3
    # keep top n word based on importance score for both stop condition evaluation and rationalization
    rational_size_ratio = None
    rational_size = 5
    # stop when target exist in top k predictions
    stop_condition_tolerance = 5

    # ======== rationalization ========
    
    approach_sample_replacing_token = "uniform"
    # approach_sample_replacing_token = "inference"
    # approach_sample_replacing_token = "postag"

    # prepare rationalizer
    if approach_sample_replacing_token == "uniform":
        # Approach 1: sample replacing token from uniform distribution
        rationalizer = AggregateRationalizer(
            importance_score_evaluator=ImportanceScoreEvaluator(
                model=model, 
                tokenizer=tokenizer, 
                token_replacer=UniformTokenReplacer(
                    token_sampler=UniformTokenSampler(tokenizer), 
                    ratio=updating_replacing_ratio
                ),
                stopping_condition_evaluator=TopKStoppingConditionEvaluator(
                    model=model, 
                    token_sampler=UniformTokenSampler(tokenizer), 
                    top_k=stop_condition_tolerance, 
                    top_n=rational_size, 
                    top_n_ratio=rational_size_ratio, 
                    tokenizer=tokenizer
                )
            ), 
            batch_size=5,
            occurrence_threshold=3,
            top_n=rational_size, 
            top_n_ratio=rational_size_ratio
        )
    elif approach_sample_replacing_token == "inference":
        # Approach 2: sample replacing token from model inference
        rationalizer = AggregateRationalizer(
            importance_score_evaluator=ImportanceScoreEvaluator(
                model=model, 
                tokenizer=tokenizer, 
                token_replacer=UniformTokenReplacer(
                    token_sampler=InferentialTokenSampler(tokenizer=tokenizer, model=model), 
                    ratio=updating_replacing_ratio
                ),
                stopping_condition_evaluator=TopKStoppingConditionEvaluator(
                    model=model, 
                    token_sampler=InferentialTokenSampler(tokenizer=tokenizer, model=model), 
                    top_k=stop_condition_tolerance, 
                    top_n=rational_size, 
                    top_n_ratio=rational_size_ratio, 
                    tokenizer=tokenizer
                )
            ), 
            batch_size=5,
            occurrence_threshold=3,
            top_n=rational_size, 
            top_n_ratio=rational_size_ratio
        )
    elif approach_sample_replacing_token == "postag":
        # Approach 3: sample replacing token from uniform distribution on a set of words with the same POS tag
        ts = POSTagTokenSampler(tokenizer=tokenizer, device=input_ids.device) # Initialize POSTagTokenSampler takes time so share it
        rationalizer = AggregateRationalizer(
            importance_score_evaluator=ImportanceScoreEvaluator(
                model=model, 
                tokenizer=tokenizer, 
                token_replacer=UniformTokenReplacer(
                    token_sampler=ts, 
                    ratio=updating_replacing_ratio
                ),
                stopping_condition_evaluator=TopKStoppingConditionEvaluator(
                    model=model, 
                    token_sampler=ts, 
                    top_k=stop_condition_tolerance, 
                    top_n=rational_size, 
                    top_n_ratio=rational_size_ratio, 
                    tokenizer=tokenizer
                )
            ), 
            batch_size=5,
            occurrence_threshold=3,
            top_n=rational_size, 
            top_n_ratio=rational_size_ratio
        )
    else:
        raise ValueError("Invalid approach_sample_replacing_token")
    
    rationalizer.trace_start()

    # rationalization
    pos_rational = rationalizer.rationalize(input_ids, generated_input[:, input_ids.shape[1]])

    # convert result (for 1st sequence in the batch)


    print()
    print(f"========================")
    print()
    print(f'Input --> {input_string[0]}')
    print(f'Target --> {tokenizer.decode(target_id[0])}')
    print(f"Rational positions --> {pos_rational}")
    print(f"Rational words -->")
    for i in range(pos_rational.shape[0]):
        ids_rational = input_ids[0, pos_rational[i]]
        text_rational = [ tokenizer.decode([id_rational]) for id_rational in ids_rational ]
        print(f"{text_rational}")

    # output

    serialize_rational(
        "rationalization_results/demo.json", 
        -1, 
        input_ids[0], 
        target_id[0], 
        pos_rational[0], 
        tokenizer, 
        rationalizer.importance_score_evaluator.important_score[0],
        compact=False,
        comments= {
            "message": "This is a demo output. [comments] is an optional field",
            "model": "gpt2-medium",
            "approach_type": approach_sample_replacing_token
        },
        trace_rationalizer=rationalizer
    )

    rationalizer.trace_stop()

if __name__ == '__main__':
    main()
