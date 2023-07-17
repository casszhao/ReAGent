
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
                        help="the saved output of rationalization log") # TODO
    parser.add_argument("--output_path", 
                        type=str,
                        default="evaluation_results/analogies/",
                        help="path to save the evaluation metrics") # TODO
    parser.add_argument("--model", 
                        type=str,
                        default="gpt2-medium",
                        help="") # TODO
    parser.add_argument("--tokenizer", 
                        type=str,
                        default="gpt2-medium",
                        help="") # TODO
    parser.add_argument("--rational-size-ratio", 
                        type=str,
                        default=0.1,
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
    output_path = args.output_path
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

    for filename in filenames:
        input_identifier = str(filename[:-5])
        path_target = os.path.join(dirpath, filename)
        with open(path_target) as f:
            rationalization_result = json.load(f)
        importance_scores = torch.tensor([rationalization_result["importance-scores"]], device=device)
        #random_importance_scores = normalise_random(torch.rand(importance_scores.size()))

        input_ids = torch.tensor([rationalization_result["input-tokens"]], device=device) # torch.Size([1, 42])
        target_id = torch.tensor([rationalization_result["target-token"]], device=device)
        print(f"==>> target_id: {target_id}")
        

        from evaluator.norm_sufficiency import Norm_Suff_Evaluator
        print(' ')
        print( ' ')
        print("  SUFF ".center(50, "-"))
        norm_suff_evaluator = Norm_Suff_Evaluator(model, rational_size_ratio)
        norm_suff = norm_suff_evaluator.evaluate(input_ids, target_id, importance_scores) # input_wte=None prob_target_original=None
        print(f"==>> norm_suff: {norm_suff}")

        # print(' ')
        # print(' ')
        # print( ' ')
        # print("  RANDOM !!! SUFF ".center(50, "-"))
        # norm_suff_random = norm_suff_evaluator.evaluate(input_ids, target_id, random_importance_scores)
        # print(f"==>> norm_suff_random: {norm_suff_random}")
        # print(' ')


        # from evaluator.soft_norm_sufficiency import Norm_Soft_Suff_Evaluator
        # print(' ')
        # print( ' ')
        # print("  SOFT SUFF ".center(50, "-"))
        # soft_norm_suff_evaluator = Norm_Soft_Suff_Evaluator(model)
        # soft_norm_suff = soft_norm_suff_evaluator.evaluate(input_ids, target_id, importance_scores)
        # print(f"==>> soft_norm_suff: {soft_norm_suff}")
        # soft_norm_suff_random = soft_norm_suff_evaluator.evaluate(input_ids, target_id, random_importance_scores)


        # from evaluator.norm_comprehensiveness import Norm_Comp_Evaluator
        # norm_comp_evaluator = Norm_Comp_Evaluator(model, rational_size_ratio)
        # norm_comp = norm_comp_evaluator.evaluate(input_ids, target_id, importance_scores)
        # norm_comp_random = norm_comp_evaluator.evaluate(input_ids, target_id, random_importance_scores)

        # from evaluator.soft_norm_comprehensiveness import Norm_Soft_Comp_Evaluator
        # soft_norm_comp_evaluator = Norm_Soft_Comp_Evaluator(model)
        # soft_norm_comp = soft_norm_comp_evaluator.evaluate(input_ids, target_id, importance_scores)
        # soft_norm_comp_random = soft_norm_comp_evaluator.evaluate(input_ids, target_id, random_importance_scores)



        logging.info(f"{filename} - {norm_suff.item()}, {soft_norm_suff.item()}, {norm_comp.item()}, {soft_norm_comp.item()}, {norm_suff_random.item()}, {soft_norm_suff_random.item()}, {norm_comp_random.item()}, {soft_norm_comp_random.item()}")
        metrics.append([input_identifier, norm_suff.item(), soft_norm_suff.item(), norm_comp.item(), soft_norm_comp.item(),
                        norm_suff_random.item(), soft_norm_suff_random.item(), norm_comp_random.item(), soft_norm_comp_random.item()])
    
    with open(f"{output_path}_{model}_details.csv", "w", newline="") as csv_f:
        writer = csv.writer(csv_f, delimiter=",", quotechar="\"", quoting=csv.QUOTE_MINIMAL)
        writer.writerow([ "id", "norm_suff", "soft_norm_suff", "norm_comp", "soft_norm_comp", 
                                "norm_suff_random", "soft_norm_suff_random", "norm_comp_random", "soft_norm_comp_random" ])
        writer.writerow([ metrics_mean[0].item(), metrics_mean[1].item(), metrics_mean[2].item(), metrics_mean[3].item(),metrics_mean[4].item(), 
                                                    metrics_mean[5].item(), metrics_mean[6].item(), metrics_mean[7].item(), metrics_mean[8].item()])

    
    print('metrics',metrics)
    metrics_rm_id = [sublist[1:] for sublist in metrics]
    print(f"==>> metrics: {metrics_rm_id}")
    metrics_t = torch.tensor(metrics_rm_id)
    metrics_mean = torch.mean(metrics_t, dim=0)

    logging.info(f"mean - {metrics_mean[0].item()}, {metrics_mean[1].item()}, {metrics_mean[2].item()}, {metrics_mean[3].item()}, {metrics_mean[4].item()}, {metrics_mean[5].item()}, {metrics_mean[6].item()}, {metrics_mean[7].item()}")

    with open(f"{output_path}_{model}.csv", "w", newline="") as csv_f2:
        writer = csv.writer(csv_f2, delimiter=",", quotechar="\"", quoting=csv.QUOTE_MINIMAL)
        writer.writerow([ "norm_suff", "soft_norm_suff", "norm_comp", "soft_norm_comp", "norm_suff_random", 
                            "soft_norm_suff_random", "norm_comp_random", "soft_norm_comp_random"])
        writer.writerow([ metrics_mean[0].item(), metrics_mean[1].item(), metrics_mean[2].item(), metrics_mean[3].item(), 
                            metrics_mean[4].item(), metrics_mean[5].item(), metrics_mean[6].item(), metrics_mean[7].item()])


if __name__ == "__main__":
    main()
