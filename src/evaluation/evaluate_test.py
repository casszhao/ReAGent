
import argparse
import csv
import json
import logging
import os
import sys
import pathlib
import torch

from transformers import AutoModelForCausalLM

from natsort import natsorted

@torch.no_grad()
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--importance_results_dir", 
                        type=str,
                        default="rationalization_results/analogies/OPT6B_integrated", # OPT6B_integrated",  #ours/top5_replace0.3_max5000_batch8", # gpt_integrated
                        help="path for storing the importance scores extracted") # TODO
    parser.add_argument("--eva_output_dir", 
                        type=str,
                        default="evaluation_results/analogies/gpt2_integrated",
                        help="") # TODO
    parser.add_argument("--model", 
                        type=str,
                        default="gpt2-medium",  # Use gpt2-medium as default for quick testing, pass any other model by commandline/job-script.
                        help="") # TODO
    parser.add_argument("--tokenizer", 
                        type=str,
                        default="gpt2-medium",  
                        help="") # TODO
    parser.add_argument("--rational_size_ratio", 
                        type=float,
                        default=1.0,  # soft then using 1
                        help="") # when using bash, it has error by cass
    parser.add_argument("--rational_size_file", 
                        type=str,
                        default=None,
                        help="A file that containing a json obj that maps sample-name to rational-size; rational_size_ratio will be ignored")
    parser.add_argument("--device", 
                        type=str,
                        default="cuda",
                        help="") # TODO
    
    parser.add_argument("--logfolder", 
                        type=str,
                        default=None,
                        help="./logs/testing_suff_comp")
    parser.add_argument("--loglevel", 
                        type=int,
                        default=20,
                        help="Debug level from [CRITICAL = 50, ERROR = 40, WARNING = 30, INFO = 20, DEBUG = 10, NOTSET = 0]")
    parser.add_argument("--cache_dir", 
                        type=str,
                        default=None,
                        help="store models")
    args = parser.parse_args()

    print(' RATIONALE RATIO ==> ', args.rational_size_ratio)

    loglevel = args.loglevel
    # setup logging system
    logger = logging.getLogger()
    logger.setLevel(loglevel)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    
    if args.logfolder:
        if not os.path.exists(args.logfolder): 
            print(' no such parent folder, create one: ', args.logfolder)
            os.makedirs(args.logfolder) 

        file_handler = logging.FileHandler(args.logfolder + 'eva.log')
        file_handler.setLevel(loglevel)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(loglevel)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    target_dir = args.importance_results_dir
    print(' target_dir: ', target_dir)
    output_dir = args.eva_output_dir
    rational_size_ratio = args.rational_size_ratio
    rational_size_file = args.rational_size_file
    device = args.device

    if args.rational_size_file != None:
        with open(args.rational_size_file) as f:
            rational_size_dict = json.load(f)

    torch.set_default_dtype(torch.float64)

    logging.info(f"Loading model...")
    model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir=args.cache_dir).to(device)
    logging.info(f"Model loaded")
    

    dirpath, dirnames, filenames = next(os.walk(target_dir))
    # filenames.sort()
    filenames = natsorted(filenames)[:2]
    # print(f"==>> {filenames}")
    # file_names = file_names[:5]

    normalise_random = torch.nn.Softmax(dim=1)

    metrics = []

    with open(os.path.join(output_dir, f'details_{args.rational_size_ratio}.csv'), "w", newline="") as csv_details_f:
        details_writer = csv.writer(csv_details_f, delimiter=",", quotechar="\"", quoting=csv.QUOTE_MINIMAL)
        details_writer.writerow(['id', "suff", "comp","random_suff", "random_comp"])
        csv_details_f.flush()

        for rational_size_ratio in [1.0]: #, 0.05, 0.1, 0.2, 0.3]:
            print(' ')
            print(' ============  rational_size_ratio  ========= ')
            print(rational_size_ratio)

            for filename in filenames:
                rational_size = -1
                # if args.rational_size_file != None:
                #     rational_size = rational_size_dict[filename]
                try: rational_size = rational_size_dict[filename]
                except: pass

                path_target = os.path.join(dirpath, filename)
                with open(path_target) as f:
                    rationalization_result = json.load(f)

                identifier = rationalization_result["id"]
                input_ids = torch.tensor([rationalization_result["input-tokens"]], device=device)
                target_id = torch.tensor([rationalization_result["target-token"]], device=device)
                importance_scores = torch.tensor([rationalization_result["importance-scores"]], device=device)
                random_importance_scores = normalise_random(torch.rand(importance_scores.size(), device=device))


                if rational_size_ratio < 1:

                    from evaluator.norm_sufficiency import NormalizedSufficiencyEvaluator
                    norm_suff_evaluator = NormalizedSufficiencyEvaluator(model, rational_size, rational_size_ratio)
                    norm_suff = norm_suff_evaluator.evaluate(input_ids, target_id, importance_scores)
                    random_norm_suff = norm_suff_evaluator.evaluate(input_ids, target_id, random_importance_scores)

                    from evaluator.norm_comprehensiveness import NormalizedComprehensivenessEvaluator
                    norm_comp_evaluator = NormalizedComprehensivenessEvaluator(model, rational_size, rational_size_ratio)
                    norm_comp = norm_comp_evaluator.evaluate(input_ids, target_id, importance_scores)
                    random_norm_comp = norm_comp_evaluator.evaluate(input_ids, target_id, random_importance_scores)

                elif rational_size_ratio == 1.0: # eva soft

                    print(' ')
                    

                    from evaluator.soft_norm_sufficiency import SoftNormalizedSufficiencyEvaluator
                    soft_norm_suff_evaluator = SoftNormalizedSufficiencyEvaluator(model)
                    norm_suff = soft_norm_suff_evaluator.evaluate(input_ids, target_id, importance_scores)
                    random_norm_suff = soft_norm_suff_evaluator.evaluate(input_ids, target_id, random_importance_scores)
                    
                    from evaluator.soft_norm_comprehensiveness import SoftNormalizedComprehensivenessEvaluator
                    soft_norm_comp_evaluator = SoftNormalizedComprehensivenessEvaluator(model)
                    norm_comp = soft_norm_comp_evaluator.evaluate(input_ids, target_id, importance_scores)
                    random_norm_comp = soft_norm_comp_evaluator.evaluate(input_ids, target_id, random_importance_scores)

                    print(f"==>> norm suff {norm_suff}, {random_norm_suff}")
                    print(f"==>> norm comp {norm_comp}, {random_norm_comp}")
                
                else: print(' args.rational_size_ratio need to be re defined between 0 to 1. 1 for soft')

                logging.info(f"{filename}:  suff & rand: {norm_suff.item()}, {random_norm_suff.item()}, comp & rand comp: {norm_comp.item()}, {random_norm_comp.item()}")
            
                metric = [identifier, norm_suff.item(), norm_comp.item(), random_norm_suff.item(), random_norm_comp.item()]
                metrics.append(metric)

                details_writer.writerow(metric)
                details_writer.writerow(metric)
                csv_details_f.flush()

            metrics_rm_id = [sublist[1:] for sublist in metrics]
            metrics_t = torch.tensor(metrics_rm_id)
            metrics_mean = torch.mean(metrics_t, dim=0)

            logging.info(f"mean : {metrics_mean[0].item()}, {metrics_mean[1].item()}, {metrics_mean[2].item()}, {metrics_mean[3].item()}")

            with open(os.path.join(output_dir, f'mean_{rational_size_ratio}.csv'), "w", newline="") as csv_mean_f:
                print(' saving mean value')
                writer = csv.writer(csv_mean_f, delimiter=",", quotechar="\"", quoting=csv.QUOTE_MINIMAL)
                writer.writerow([ "suff", "comp", "random_suff", "random_comp"])
                writer.writerow([ metrics_mean[0].item(), metrics_mean[1].item(), metrics_mean[2].item(), metrics_mean[3].item()])
                print("suff", metrics_mean[0].item(), "comp", metrics_mean[1].item())
                print("random_suff", metrics_mean[2].item(), "random_comp",  metrics_mean[3].item())
                print(' final suff :', metrics_mean[0].item()/metrics_mean[2].item())
                print(' final comp :', metrics_mean[1].item()/metrics_mean[3].item())
if __name__ == "__main__":
    main()
