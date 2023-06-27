import argparse
import json
import logging
import os
import sys
import time

import torch
from rationalizer.aggregate_rationalizer import AggregateRationalizer
from rationalizer.importance_score_evaluator.delta_prob import DeltaProbImportanceScoreEvaluator
from rationalizer.importance_score_evaluator.bayesian_opti import BayesianOptimizationImportanceScoreEvaluator
from rationalizer.sample_rationalizer import SampleRationalizer
from rationalizer.stopping_condition_evaluator.top_k import TopKStoppingConditionEvaluator
from rationalizer.token_replacement.token_replacer.uniform import UniformTokenReplacer
from rationalizer.token_replacement.token_replacer.ranking import RankingTokenReplacer
from rationalizer.token_replacement.token_sampler.inferential import InferentialTokenSampler
from rationalizer.token_replacement.token_sampler.postag import POSTagTokenSampler
from rationalizer.token_replacement.token_sampler.uniform import UniformTokenSampler
from rationalizer.utils.serializing import serialize_rational
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", 
                        type=str,
                        default="gpt2", # gpt2-medium gpt2-large
                        help="") # TODO
    parser.add_argument("--tokenizer", 
                        type=str,
                        default="gpt2",
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

    parser.add_argument("--rationalization-config", 
                        type=str,
                        default="config/test.json",
                        help="") # TODO
    
    parser.add_argument("--logfile", 
                        type=str,
                        default=None,
                        help="Logfile location to output")
    
    parser.add_argument("--input_data_size", 
                        type=float,
                        default=1,
                        help="how many data we want to feed to model (percentage)")
    args = parser.parse_args()

    # setup logging system
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) # DEBUG OR INFO
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    
    if args.logfile:
        file_handler = logging.FileHandler(args.logfile)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    data_dir = args.data_dir
    output_dir = args.output_dir
    device = args.device
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)

    with open(args.rationalization_config) as f_config:
        rationalization_config = json.load(f_config)

    replacing_type = rationalization_config["replacing"]["type"]
    if replacing_type == "uniform":
        token_sampler = UniformTokenSampler(tokenizer)
    elif replacing_type == "inferential":
        token_sampler = InferentialTokenSampler(tokenizer=tokenizer, model=model)
    elif replacing_type == "postag":
        token_sampler = POSTagTokenSampler(tokenizer=tokenizer, device=device)
    else:
        raise ValueError(f"Invalid replacement_sampling: {replacing_type}")
    
    rationalizer_type = rationalization_config["rationalizer"]["type"]
    evaluator_type = rationalization_config["importance_score_evaluator"]["type"]
    if rationalizer_type == "sampling":
        if evaluator_type == 'delta_probability':
            importance_score_evaluator = DeltaProbImportanceScoreEvaluator(
                model=model, 
                tokenizer=tokenizer, 
                token_replacer=UniformTokenReplacer(
                    token_sampler=token_sampler, 
                    ratio=rationalization_config["importance_score_evaluator"]["delta_probability"]["replacing_ratio"]
                ),
                stopping_condition_evaluator=TopKStoppingConditionEvaluator(
                    model=model, 
                    token_sampler=token_sampler, 
                    top_k=rationalization_config["stopping_condition"]["top_k"]["tolerance"], 
                    top_n=rationalization_config["rational"]["size"], 
                    top_n_ratio=rationalization_config["rational"]["size_ratio"], 
                    tokenizer=tokenizer
                )
            )
        elif evaluator_type == 'bayesian_optimization':
            importance_score_evaluator = BayesianOptimizationImportanceScoreEvaluator(
                model=model, 
                tokenizer=tokenizer, 
                token_replacer=RankingTokenReplacer(
                    token_sampler=token_sampler, 
                    top_n=rationalization_config["rational"]["size"], 
                    top_n_ratio=rationalization_config["rational"]["size_ratio"], 
                ),
                stopping_condition_evaluator=TopKStoppingConditionEvaluator(
                    model=model, 
                    token_sampler=token_sampler, 
                    top_k=rationalization_config["stopping_condition"]["top_k"]["tolerance"], 
                    top_n=rationalization_config["rational"]["size"], 
                    top_n_ratio=rationalization_config["rational"]["size_ratio"], 
                    tokenizer=tokenizer
                ),
                sample_multiplier=rationalization_config["importance_score_evaluator"]["bayesian_optimization"]["sampling"]["multiplier"],
                sample_increment=rationalization_config["importance_score_evaluator"]["bayesian_optimization"]["sampling"]["increment"],
                training_config=rationalization_config["importance_score_evaluator"]["bayesian_optimization"]["training"],
                optimizing_config=rationalization_config["importance_score_evaluator"]["bayesian_optimization"]["optimizing"]
            )
        else:
            raise ValueError(f"Invalid evaluator-type: {evaluator_type}")
        rationalizer = SampleRationalizer(
            importance_score_evaluator=importance_score_evaluator, 
            top_n=rationalization_config["rational"]["size"], 
            top_n_ratio=rationalization_config["rational"]["size_ratio"]
        )
    elif rationalizer_type == "aggregation":
        assert evaluator_type == 'delta_probability', 'aggregate rationalizer require delta_probability evaluator'

        rationalizer = AggregateRationalizer(
            importance_score_evaluator=DeltaProbImportanceScoreEvaluator(
                model=model, 
                tokenizer=tokenizer, 
                token_replacer=UniformTokenReplacer(
                    token_sampler=token_sampler, 
                    ratio=rationalization_config["importance_score_evaluator"]["delta_probability"]["replacing_ratio"]
                ),
                stopping_condition_evaluator=TopKStoppingConditionEvaluator(
                    model=model, 
                    token_sampler=token_sampler, 
                    top_k=rationalization_config["stopping_condition"]["top_k"]["tolerance"], 
                    top_n=rationalization_config["rational"]["size"], 
                    top_n_ratio=rationalization_config["rational"]["size_ratio"], 
                    tokenizer=tokenizer
                )
            ),
            batch_size=rationalization_config["rationalizer"]["aggregation"]["batch_size"],
            overlap_threshold=rationalization_config["rationalizer"]["aggregation"]["overlap_threshold"],
            overlap_strict_pos=rationalization_config["rationalizer"]["aggregation"]["overlap_strict_pos"],
            top_n=rationalization_config["rational"]["size"], 
            top_n_ratio=rationalization_config["rational"]["size_ratio"]
        )
    else:
        raise ValueError(f"Invalid rationalizer_type {rationalizer_type}")
    
    dirpath, dirnames, filenames = next(os.walk(data_dir))
    filenames.sort()

    total_file_num = len(filenames)

    if args.input_data_size != 1:
        input_num = int(total_file_num * args.input_data_size)
        if input_num < 1: input_num = 1
        filenames = filenames[:input_num]
    

    # run all experiments
    
    for filename in filenames:

        with open(os.path.join(dirpath, filename)) as data_f:
            data = json.load(data_f)

        tokens = torch.unsqueeze(torch.tensor(data['tokens'], device=device), 0)
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
