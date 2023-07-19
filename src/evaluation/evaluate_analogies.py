
import argparse
import csv
import json
import logging
import os
import sys

import torch

from transformers import AutoModelForCausalLM

from natsort import natsorted

@torch.no_grad()
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--target-dir", 
                        type=str,
                        default="rationalization_results/analogies/gpt2-medium.sampling.uniform",
                        help="") # TODO
    parser.add_argument("--output-dir", 
                        type=str,
                        default="evaluation_results/analogies/",
                        help="") # TODO
    parser.add_argument("--model", 
                        type=str,
                        default="bigscience/bloom",  # KoboldAI/OPT-6.7B-Erebus openlm-research/open_llama_7b_v2
                        help="") # TODO
    parser.add_argument("--tokenizer", 
                        type=str,
                        default="bigscience/bloom",  
                        help="") # TODO
    parser.add_argument("--rational-size-ratio", 
                        type=str,
                        default=0.3,
                        help="") # TODO
    parser.add_argument("--device", 
                        type=str,
                        default="cuda",
                        help="") # TODO
    
    parser.add_argument("--logfile", 
                        type=str,
                        default=None,
                        help="Logfile location to output")
    parser.add_argument("--loglevel", 
                        type=int,
                        default=20,
                        help="Debug level from [CRITICAL = 50, ERROR = 40, WARNING = 30, INFO = 20, DEBUG = 10, NOTSET = 0]")
    args = parser.parse_args()

    loglevel = args.loglevel
    # setup logging system
    logger = logging.getLogger()
    logger.setLevel(loglevel)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    
    if args.logfile:
        file_handler = logging.FileHandler(args.logfile)
        file_handler.setLevel(loglevel)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(loglevel)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    target_dir = args.target_dir
    output_dir = args.output_dir
    rational_size_ratio = args.rational_size_ratio
    device = args.device

    torch.set_default_dtype(torch.float64)

    logging.info(f"Loading model...")
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    logging.info(f"Model loaded")
    

    dirpath, dirnames, filenames = next(os.walk(target_dir))
    # filenames.sort()
    filenames = natsorted(filenames)

    normalise_random = torch.nn.Softmax(dim=1)

    metrics = []

    with open(os.path.join(output_dir, 'details.csv'), "w", newline="") as csv_details_f:
        details_writer = csv.writer(csv_details_f, delimiter=",", quotechar="\"", quoting=csv.QUOTE_MINIMAL)
        details_writer.writerow(['id', "norm_suff", "soft_norm_suff", "norm_comp", "soft_norm_comp","random_norm_suff", "random_soft_norm_suff", "random_norm_comp", "random_soft_norm_comp"])
        csv_details_f.flush()

        for filename in filenames:
            path_target = os.path.join(dirpath, filename)
            with open(path_target) as f:
                rationalization_result = json.load(f)

            identifier = rationalization_result["id"]
            input_ids = torch.tensor([rationalization_result["input-tokens"]], device=device)
            target_id = torch.tensor([rationalization_result["target-token"]], device=device)
            importance_scores = torch.tensor([rationalization_result["importance-scores"]], device=device)
            random_importance_scores = normalise_random(torch.rand(importance_scores.size(), device=device))

            from evaluator.norm_sufficiency import NormalizedSufficiencyEvaluator
            norm_suff_evaluator = NormalizedSufficiencyEvaluator(model, rational_size_ratio)
            norm_suff = norm_suff_evaluator.evaluate(input_ids, target_id, importance_scores)
            random_norm_suff = norm_suff_evaluator.evaluate(input_ids, target_id, random_importance_scores)

            from evaluator.norm_comprehensiveness import NormalizedComprehensivenessEvaluator
            norm_comp_evaluator = NormalizedComprehensivenessEvaluator(model, rational_size_ratio)
            norm_comp = norm_comp_evaluator.evaluate(input_ids, target_id, importance_scores)
            random_norm_comp = norm_comp_evaluator.evaluate(input_ids, target_id, random_importance_scores)

            from evaluator.soft_norm_sufficiency import SoftNormalizedSufficiencyEvaluator
            soft_norm_suff_evaluator = SoftNormalizedSufficiencyEvaluator(model)
            soft_norm_suff = soft_norm_suff_evaluator.evaluate(input_ids, target_id, importance_scores)
            random_soft_norm_suff = soft_norm_suff_evaluator.evaluate(input_ids, target_id, random_importance_scores)
            
            from evaluator.soft_norm_comprehensiveness import SoftNormalizedComprehensivenessEvaluator
            soft_norm_comp_evaluator = SoftNormalizedComprehensivenessEvaluator(model)
            soft_norm_comp = soft_norm_comp_evaluator.evaluate(input_ids, target_id, importance_scores)
            random_soft_norm_comp = soft_norm_comp_evaluator.evaluate(input_ids, target_id, random_importance_scores)

            logging.info(f"{filename} - {norm_suff.item()}, {soft_norm_suff.item()}, {norm_comp.item()}, {soft_norm_comp.item()}, {random_norm_suff.item()}, {random_soft_norm_suff.item()}, {random_norm_comp.item()}, {random_soft_norm_comp.item()}")
            metric = [identifier,
                      norm_suff.item(), soft_norm_suff.item(), norm_comp.item(), soft_norm_comp.item(),
                      random_norm_suff.item(), random_soft_norm_suff.item(), random_norm_comp.item(), random_soft_norm_comp.item()]
            metrics.append(metric)

            details_writer.writerow(metric)
            csv_details_f.flush()

    metrics_rm_id = [sublist[1:] for sublist in metrics]
    metrics_t = torch.tensor(metrics_rm_id)
    metrics_mean = torch.mean(metrics_t, dim=0)

    logging.info(f"mean - {metrics_mean[0].item()}, {metrics_mean[1].item()}, {metrics_mean[2].item()}, {metrics_mean[3].item()}, {metrics_mean[4].item()}, {metrics_mean[5].item()}, {metrics_mean[6].item()}, {metrics_mean[7].item()}")

    with open(os.path.join(output_dir, 'mean.csv'), "w", newline="") as csv_mean_f:
        writer = csv.writer(csv_mean_f, delimiter=",", quotechar="\"", quoting=csv.QUOTE_MINIMAL)
        writer.writerow([ "norm_suff", "soft_norm_suff", "norm_comp", "soft_norm_comp","random_norm_suff", "random_soft_norm_suff", "random_norm_comp", "random_soft_norm_comp" ])
        writer.writerow([ metrics_mean[0].item(), metrics_mean[1].item(), metrics_mean[2].item(), metrics_mean[3].item(), metrics_mean[4].item(), metrics_mean[5].item(), metrics_mean[6].item(), metrics_mean[7].item() ])

if __name__ == "__main__":
    main()
