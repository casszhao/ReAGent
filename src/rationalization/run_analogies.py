import argparse
import json
import logging
import os
import sys
import time

import torch
from natsort import natsorted
from rationalizer.utils.serializing import serialize_rational
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", 
                        type=str,
                        default="gpt2-medium", # gpt2-medium gpt2-large
                        help="") # TODO
    parser.add_argument("--cache_dir", 
                        type=str,
                        default='cache/',
                        help="store models")
    parser.add_argument("--tokenizer", 
                        type=str,
                        default="gpt2-medium",
                        help="") # TODO
    parser.add_argument("--data-dir", 
                        type=str,
                        default="data/analogies/gpt2",
                        help="") # TODO
    parser.add_argument("--importance_results_dir", 
                        type=str,
                        default="rationalization_results/analogies/test",
                        help="") # TODO
    parser.add_argument("--device", 
                        type=str,
                        default="cuda",
                        help="") # TODO

    parser.add_argument("--rationalization-config", 
                        type=str,
                        default="config/test.json",
                        help="") # TODO

    parser.add_argument("--input_num_ratio", 
                        type=float,
                        default=1,
                        help="") # TODO
    
    parser.add_argument("--seed", 
                        type=int,
                        default=42,
                        help="Random seed")

    
    parser.add_argument("--logfolder", 
                        type=str,
                        default=None,
                        help="Logfile location to output")
    parser.add_argument("--loglevel", 
                        type=int,
                        default=20,
                        help="Debug level from [CRITICAL = 50, ERROR = 40, WARNING = 30, INFO = 20, DEBUG = 10, NOTSET = 0]")
    args = parser.parse_args()

    rand_seed: int = args.seed
    if rand_seed:
        torch.manual_seed(rand_seed)
        torch.use_deterministic_algorithms(True, warn_only=True)

    loglevel = args.loglevel
    # setup logging system
    logger = logging.getLogger()
    logger.setLevel(loglevel)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    
    if args.logfolder:
        if not os.path.exists(args.logfolder): 
            print(' no such parent folder, create one: ', args.logfolder)
            os.makedirs(args.logfolder) 

        file_handler = logging.FileHandler(args.logfolder + 'run.log')
        file_handler.setLevel(loglevel)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(loglevel)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    data_dir = args.data_dir
    output_dir = args.importance_results_dir
    device = args.device
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir=args.cache_dir).to(device)

    with open(args.rationalization_config) as f_config:
        rationalization_config = json.load(f_config)

    importance_score_evaluator_type = rationalization_config["importance_score_evaluator"]["type"]

    if importance_score_evaluator_type == "replacing":

        replacing_type = rationalization_config["importance_score_evaluator"]["replacing"]["replacing"]["type"]
        if replacing_type == "uniform":
            from rationalizer.token_replacement.token_sampler.uniform import \
                UniformTokenSampler
            token_sampler = UniformTokenSampler(tokenizer)
        elif replacing_type == "inferential":
            from rationalizer.token_replacement.token_sampler.inferential import \
                InferentialTokenSampler
            token_sampler = InferentialTokenSampler(tokenizer=tokenizer, model=model)
        elif replacing_type == "postag":
            from rationalizer.token_replacement.token_sampler.postag import \
                POSTagTokenSampler
            token_sampler = POSTagTokenSampler(tokenizer=tokenizer, device=device)
        elif replacing_type == "inferential-m":
            from rationalizer.token_replacement.token_sampler.inferential_m import \
                InferentialMTokenSampler
            from transformers import AutoModelForMaskedLM
            sampler_tokenizer = AutoTokenizer.from_pretrained(
                rationalization_config["importance_score_evaluator"]["replacing"]["replacing"]["inferential-m"]["tokenizer"], 
                cache_dir=args.cache_dir)
            sampler_model = AutoModelForMaskedLM.from_pretrained(
                rationalization_config["importance_score_evaluator"]["replacing"]["replacing"]["inferential-m"]["model"], 
                cache_dir=args.cache_dir).to(device)
            token_sampler = InferentialMTokenSampler(
                source_tokenizer=tokenizer,
                sampler_tokenizer=sampler_tokenizer,
                sampler_model=sampler_model)
        else:
            raise ValueError(f"Invalid replacement_sampling: {replacing_type}")
        
        stopping_condition_type = rationalization_config["importance_score_evaluator"]["replacing"]["stopping_condition"]["type"]
        if stopping_condition_type == "top_k":
            from rationalizer.stopping_condition_evaluator.top_k import \
                TopKStoppingConditionEvaluator
            top_k=rationalization_config["importance_score_evaluator"]["replacing"]["stopping_condition"]["top_k"]["tolerance"]
            top_n=rationalization_config["rational"]["size"]
            top_n_ratio=rationalization_config["rational"]["size_ratio"]
            stopping_condition_evaluator = TopKStoppingConditionEvaluator(
                model=model, 
                token_sampler=token_sampler, 
                top_k=top_k, 
                top_n=top_n, 
                top_n_ratio=top_n_ratio, 
                tokenizer=tokenizer
            )
            #output_dir = output_dir + f'/top{top_k}_' # by cass

        elif stopping_condition_type == "dummy":
            from rationalizer.stopping_condition_evaluator.dummy import \
                DummyStoppingConditionEvaluator
            stopping_condition_evaluator = DummyStoppingConditionEvaluator()
        else:
            raise ValueError(f"Invalid stopping_condition: {stopping_condition_type}")

        evaluator_type = rationalization_config["importance_score_evaluator"]["replacing"]["optimization"]["type"]
        if evaluator_type == 'delta_probability':
            from rationalizer.importance_score_evaluator.delta_prob import \
                DeltaProbImportanceScoreEvaluator
            from rationalizer.token_replacement.token_replacer.uniform import \
                UniformTokenReplacer
            replacing_ratio=rationalization_config["importance_score_evaluator"]["replacing"]["optimization"]["delta_probability"]["replacing_ratio"]
            max_steps=rationalization_config["importance_score_evaluator"]["replacing"]["optimization"]["delta_probability"]["max_steps"]
            # output_dir = output_dir + f'replace{replacing_ratio}_max{max_steps}' # by cass
            importance_score_evaluator = DeltaProbImportanceScoreEvaluator(
                model=model, 
                tokenizer=tokenizer, 
                token_replacer=UniformTokenReplacer(
                    token_sampler=token_sampler, 
                    ratio=replacing_ratio
                ),
                stopping_condition_evaluator=stopping_condition_evaluator,
                max_steps=max_steps
            )
        elif evaluator_type == 'bayesian_optimization':
            from rationalizer.importance_score_evaluator.bayesian_opti import \
                BayesianOptimizationImportanceScoreEvaluator
            from rationalizer.token_replacement.token_replacer.ranking import \
                RankingTokenReplacer
            importance_score_evaluator = BayesianOptimizationImportanceScoreEvaluator(
                model=model, 
                tokenizer=tokenizer, 
                token_replacer=RankingTokenReplacer(
                    token_sampler=token_sampler, 
                    top_n=rationalization_config["rational"]["size"], 
                    top_n_ratio=rationalization_config["rational"]["size_ratio"], 
                ),
                stopping_condition_evaluator=stopping_condition_evaluator,
                sample_multiplier=rationalization_config["importance_score_evaluator"]["replacing"]["optimization"]["bayesian_optimization"]["sampling"]["multiplier"],
                sample_increment=rationalization_config["importance_score_evaluator"]["replacing"]["optimization"]["bayesian_optimization"]["sampling"]["increment"],
                training_config=rationalization_config["importance_score_evaluator"]["replacing"]["optimization"]["bayesian_optimization"]["training"],
                optimizing_config=rationalization_config["importance_score_evaluator"]["replacing"]["optimization"]["bayesian_optimization"]["optimizing"]
            )
        else:
            raise ValueError(f"Invalid evaluator-type: {evaluator_type}")
    
    elif importance_score_evaluator_type == "attention":
        from rationalizer.importance_score_evaluator.attention import \
            AttentionImportanceScoreEvaluator
        importance_score_evaluator = AttentionImportanceScoreEvaluator(
            model=model,
            tokenizer=tokenizer,
            attn_type=rationalization_config["importance_score_evaluator"]["attention"]["type"]
        )
    elif importance_score_evaluator_type == "gradient":
        from rationalizer.importance_score_evaluator.grad import \
            GradientImportanceScoreEvaluator
        importance_score_evaluator = GradientImportanceScoreEvaluator(
            model=model,
            tokenizer=tokenizer,
            grad_type=rationalization_config["importance_score_evaluator"]["gradient"]["type"]
        )
    elif importance_score_evaluator_type == "inseq":
        from rationalizer.importance_score_evaluator.inseq import \
            InseqImportanceScoreEvaluator
        importance_score_evaluator = InseqImportanceScoreEvaluator(
            model=model,
            tokenizer=tokenizer,
            method=rationalization_config["importance_score_evaluator"]["inseq"]["type"],
            attribute_params=rationalization_config["importance_score_evaluator"]["inseq"]["attribute_params"]
        )
    else:
        raise ValueError(f"Invalid importance_score_evaluator_type {importance_score_evaluator_type}")
        
    rationalizer_type = rationalization_config["rationalizer"]["type"]
    if rationalizer_type == "sampling":
        from rationalizer.sample_rationalizer import SampleRationalizer
        rationalizer = SampleRationalizer(
            importance_score_evaluator=importance_score_evaluator, 
            top_n=rationalization_config["rational"]["size"], 
            top_n_ratio=rationalization_config["rational"]["size_ratio"]
        )
    elif rationalizer_type == "aggregation":
        from rationalizer.aggregate_rationalizer import AggregateRationalizer
        rationalizer = AggregateRationalizer(
            importance_score_evaluator=importance_score_evaluator,
            batch_size=rationalization_config["rationalizer"]["aggregation"]["batch_size"],
            overlap_threshold=rationalization_config["rationalizer"]["aggregation"]["overlap_threshold"],
            overlap_strict_pos=rationalization_config["rationalizer"]["aggregation"]["overlap_strict_pos"],
            top_n=rationalization_config["rational"]["size"], 
            top_n_ratio=rationalization_config["rational"]["size_ratio"]
        )
    else:
        raise ValueError(f"Invalid rationalizer_type {rationalizer_type}")
    
    dirpath, dirnames, filenames = next(os.walk(data_dir))
    # filenames.sort()
    filenames = natsorted(filenames)

    total_file_num = len(filenames)

    if args.input_num_ratio >= 1:
        pass
    else:
        input_num = int(total_file_num * args.input_num_ratio)
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
        time_start = time.time()
        pos_rational = rationalizer.rationalize(input_tokens, target_token)
        time_end = time.time()
        time_elapsed = time_end - time_start
        logging.info(f"Rationalization done in {time_elapsed}")

        # convert results

        logging.info("")
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
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_filename = os.path.join(output_dir, filename)
        print(f"==>> output_filename (importance scroes saved to): {output_filename}")

        # TODO: Save model for BO
        
        comments = {
            "created-by": os.path.basename(__file__),
            "args" : args.__dict__,
            "time_elapsed": time_elapsed
        }

        # Append comment for steps of updating
        if rationalization_config["importance_score_evaluator"]["type"] == "delta_probability":
            comments["updating"] = {
                "num_steps": rationalizer.importance_score_evaluator.num_steps,
                "max_steps": rationalization_config["importance_score_evaluator"]["delta_probability"]["max_steps"]
            }

        # Append comment for separate_rational
        if rationalization_config["rationalizer"]["type"] == "aggregation" and rationalization_config["rationalizer"]["aggregation"]["save_separate_rational"]:
            pos_rationals, rationals = rationalizer.get_separate_rational(input_tokens, tokenizer)
            comments["separate_rational"] = rationals
        
        if rationalizer_type == "aggregation" and evaluator_type == 'delta_probability':
            # ignore runs that not hit the stopping contition from mean
            delta_prob_evaluator : DeltaProbImportanceScoreEvaluator = importance_score_evaluator
            stop_mask = delta_prob_evaluator.stop_mask
            important_score = rationalizer.importance_score_evaluator.important_score

            important_score_masked = important_score * torch.unsqueeze(stop_mask, -1)
            important_score_mean = torch.sum(important_score_masked, dim=0) / torch.sum(stop_mask)
        else:
            important_score_mean = torch.mean(rationalizer.importance_score_evaluator.important_score, dim=0)

        serialize_rational(
            output_filename,
            data["id"], 
            input_tokens[0], 
            target_token[0], 
            pos_rational[0], 
            tokenizer, 
            important_score_mean,
            compact=False,
            comments=comments,
            # trace_rationalizer=rationalizer # Enable trace logs
        )
        
        # rationalizer.trace_stop()

        logging.info(f'{filename} done.')
        logging.info("")
        logging.info(f"========================")
        logging.info("")

if __name__ == "__main__":
    main()
