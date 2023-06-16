import argparse
import json
import logging
import os
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
from aggregate_rationalizer import AggregateRationalizer
from utils.serializing import serialize_rational
from token_replacement.token_replacer.uniform import UniformTokenReplacer
from importance_score_evaluator import ImportanceScoreEvaluator
from rationalizer import Rationalizer
from stopping_condition_evaluator.top_k import TopKStoppingConditionEvaluator
from token_replacement.token_sampler.inferential import InferentialTokenSampler
from token_replacement.token_sampler.postag import POSTagTokenSampler
from token_replacement.token_sampler.uniform import UniformTokenSampler


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--replacement-sampling", 
                        type=str,
                        default="uniform",
                        help="[uniform, inferential, postag]")
    parser.add_argument("--rationalizer", 
                        type=str,
                        default="sampling",
                        help="[sampling, aggregate]") # TODO
    parser.add_argument("--model", 
                        type=str,
                        default="gpt2-medium",
                        help="") # TODO
    parser.add_argument("--tokenizer", 
                        type=str,
                        default="gpt2-medium",
                        help="") # TODO
    parser.add_argument("--data-dir", 
                        type=str,
                        default="data/analogies",
                        help="") # TODO
    parser.add_argument("--output-dir", 
                        type=str,
                        default="rationalization_results/analogies/gpt2-medium.sampling.uniform",
                        help="") # TODO
    parser.add_argument("--device", 
                        type=str,
                        default="cuda",
                        help="") # TODO

    parser.add_argument("--updating-replacing-ratio", 
                        type=float,
                        default=0.3,
                        help="replacing ratio during importance score updating")
    parser.add_argument("--rational-size-ratio", 
                        type=float,
                        default=None,
                        help="keep top n word based on importance score for both stop condition evaluation and rationalization")
    parser.add_argument("--rational-size", 
                        type=int,
                        default=5,
                        help="keep top n word based on importance score for both stop condition evaluation and rationalization")
    parser.add_argument("--stopping-condition-tolerance", 
                        type=int,
                        default=5,
                        help="stop when target exist in top k predictions")
    parser.add_argument("--aggregate-batch-size", 
                        type=int,
                        default=5,
                        help="Batch size for aggregation")
    parser.add_argument("--aggregate-overlap-threshold", 
                        type=int,
                        default=3,
                        help="Overlap threshold of rational tokens within a batch")
    parser.add_argument("--aggregate-overlap-strict-pos", 
                        type=bool,
                        default=True,
                        help="Whether overlap strict to position ot not")
    args = parser.parse_args()


    replacement_sampling_type = args.replacement_sampling
    rationalizer_type = args.rationalizer
    data_dir = args.data_dir
    output_dir = args.output_dir
    device = args.device

    # replacing ratio during importance score updating
    updating_replacing_ratio = args.updating_replacing_ratio
    # keep top n word based on importance score for both stop condition evaluation and rationalization
    rational_size_ratio = args.rational_size_ratio
    rational_size = args.rational_size
    # stop when target exist in top k predictions
    stopping_condition_tolerance = args.stopping_condition_tolerance

    # Batch size for aggregate
    aggregate_batch_size = args.aggregate_batch_size
    # Overlap threshold of rational tokens within a batch
    overlap_threshold = args.aggregate_overlap_threshold
    # Whether overlap strict to position ot not
    overlap_strict_pos = args.aggregate_overlap_strict_pos

    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForCausalLM.from_pretrained(args.model)


    if replacement_sampling_type == "uniform":
        token_sampler = UniformTokenSampler(tokenizer)
    elif replacement_sampling_type == "inferential":
        token_sampler = InferentialTokenSampler(tokenizer=tokenizer, model=model)
    elif replacement_sampling_type == "postag":
        token_sampler = POSTagTokenSampler(tokenizer=tokenizer, device=device)
    else:
        raise ValueError(f"Invalid replacement_sampling: {replacement_sampling_type}")
    
    if rationalizer_type == "sampling":
        rationalizer = Rationalizer(
            importance_score_evaluator=ImportanceScoreEvaluator(
                model=model, 
                tokenizer=tokenizer, 
                token_replacer=UniformTokenReplacer(
                    token_sampler=token_sampler, 
                    ratio=updating_replacing_ratio
                ),
                stopping_condition_evaluator=TopKStoppingConditionEvaluator(
                    model=model, 
                    token_sampler=token_sampler, 
                    top_k=stopping_condition_tolerance, 
                    top_n=rational_size, 
                    top_n_ratio=rational_size_ratio, 
                    tokenizer=tokenizer
                )
            ), 
            top_n=rational_size, 
            top_n_ratio=rational_size_ratio
        )
    elif rationalizer_type == "aggregate":
        rationalizer = AggregateRationalizer(
            importance_score_evaluator=ImportanceScoreEvaluator(
                model=model, 
                tokenizer=tokenizer, 
                token_replacer=UniformTokenReplacer(
                    token_sampler=token_sampler, 
                    ratio=updating_replacing_ratio
                ),
                stopping_condition_evaluator=TopKStoppingConditionEvaluator(
                    model=model, 
                    token_sampler=token_sampler, 
                    top_k=stopping_condition_tolerance, 
                    top_n=rational_size, 
                    top_n_ratio=rational_size_ratio, 
                    tokenizer=tokenizer
                )
            ),
            batch_size=aggregate_batch_size,
            overlap_threshold=overlap_threshold,
            overlap_strict_pos=overlap_strict_pos,
            top_n=rational_size, 
            top_n_ratio=rational_size_ratio
        )
    
    dirpath, dirnames, filenames = next(os.walk(data_dir))
    filenames.sort()

    # run all experiments
    
    for filename in filenames:

        with open(os.path.join(dirpath, filename)) as data_f:
            data = json.load(data_f)

        tokens = torch.unsqueeze(torch.tensor(data['tokens']), 0)
        input_tokens = tokens[:, :data["target"]]
        target_token = tokens[:, data["target"]]
        logging.info(f"Rationalizing {filename} ...")

        # rationalizer.trace_start()

        # rationalization
        with torch.no_grad():
            time_start = time.time()
            pos_rational = rationalizer.rationalize(input_tokens, target_token)
            time_end = time.time()
            time_elapsed = time_end - time_start
            logging.info(f"Rationalization done in {time_elapsed}")

        # convert results

        logging.info("")
        logging.info(f"========================")
        logging.info("")
        logging.info(f'Input --> {tokenizer.decode(input_tokens[0])}')
        logging.info(f'Target --> {tokenizer.decode(target_token[0])}')
        logging.info(f"Rational positions --> {pos_rational}")
        logging.info(f"Rational words -->")
        for i in range(pos_rational.shape[0]):
            ids_rational = tokens[0, pos_rational[i]]
            text_rational = [ tokenizer.decode([id_rational]) for id_rational in ids_rational ]
            logging.info(f"{text_rational}")

        # output
        output_filename = os.path.join(output_dir, filename)
        serialize_rational(
            output_filename,
            data["id"], 
            input_tokens[0], 
            target_token[0], 
            pos_rational[0], 
            tokenizer, 
            rationalizer.importance_score_evaluator.important_score[0],
            compact=False,
            comments= {
                "created-by": os.path.basename(__file__),
                "args" : args.__dict__,
                "time_elapsed": time_elapsed
            },
            # trace_rationalizer=rationalizer # Enable trace logs
        )
        
        # rationalizer.trace_stop()

        logging.info(f'{filename} done.')
