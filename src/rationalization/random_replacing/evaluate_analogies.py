
import argparse
import csv
import json
import os

import torch

from transformers import AutoTokenizer


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", 
                        type=str,
                        default="data/analogies",
                        help="") # TODO
    parser.add_argument("--target-dir", 
                        type=str,
                        default="rationalization_results/analogies/gpt2-medium.sampling.uniform",
                        help="") # TODO
    parser.add_argument("--baseline-dir", 
                        type=str,
                        default="rationalization_results/analogies/gpt2-medium.last_attention",
                        help="") # TODO
    parser.add_argument("--output-path", 
                        type=str,
                        default="evaluation_results/analogies/gpt2-medium.sampling.uniform.csv",
                        help="") # TODO
    parser.add_argument("--tokenizer", 
                        type=str,
                        default="gpt2-medium",
                        help="") # TODO
    args = parser.parse_args()

    data_dir = args.data_dir
    target_dir = args.target_dir
    baseline_dir = args.baseline_dir
    output_path = args.output_path
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    rational_sizes = []
    no_distractors = []
    contain_relatives = []
    baseline_approximation_ratios = []

    dirpath, dirnames, filenames = next(os.walk(target_dir))
    filenames.sort()

    for filename in filenames:
        path_target = os.path.join(dirpath, filename)
        with open(path_target) as f:
            result_target = json.load(f)

        path_data = os.path.join(data_dir, filename)
        if not os.path.exists(path_data):
            print(f"[Warning] {path_data} not found. Skipping ground truth.")
        else:
            with open(path_data) as f:
                data = json.load(f)
            
            # rational_sizes
            rational_size_target = result_target["rational-size"]
            rational_sizes.append(rational_size_target)

            # no_distractors
            pos_rational = torch.tensor(result_target["rational-positions"])
            non_distractor_rational = (pos_rational < data["distractor"]["start"]) + (pos_rational > data["distractor"]["end"])
            no_distractor = torch.sum(non_distractor_rational) == rational_sizes
            no_distractors.append(no_distractor)

            # contain_relatives
            relative_hits = pos_rational == data["relative"]
            contain_relative = torch.sum(relative_hits) > 0
            contain_relatives.append(contain_relative)

        path_baseline = os.path.join(baseline_dir, filename)
        if not os.path.exists(path_baseline):
            print(f"[Warning] {path_baseline} not found. Skipping baseline.")
        else:
            with open(path_baseline) as f:
                result_baseline = json.load(f)

            rational_size_baseline = result_baseline["rational-size"]

            # baseline_approximation_ratios
            baseline_approximation_ratio = rational_size_target / rational_size_baseline
            baseline_approximation_ratios.append(baseline_approximation_ratio)
    
    mean_rational_size = torch.mean(torch.tensor(rational_sizes, dtype=torch.float))
    ratio_no_distractor = torch.mean(torch.tensor(no_distractors, dtype=torch.float))
    ratio_contain_relative = torch.mean(torch.tensor(contain_relatives, dtype=torch.float))
    mean_baseline_approximation_ratio = torch.mean(torch.tensor(baseline_approximation_ratios, dtype=torch.float))

    print(f"Mean rational size: {mean_rational_size}")
    print(f"Ratio no distractor: {ratio_no_distractor}")
    print(f"Ratio contain relative: {ratio_contain_relative}")
    print(f"Mean baseline approximation ratio: {mean_baseline_approximation_ratio}")

    with open(output_path, "w", newline="") as csv_f:
        writer = csv.writer(csv_f, delimiter=",", quotechar="\"", quoting=csv.QUOTE_MINIMAL)
        writer.writerow([ "Mean rational size", "Ratio no distractor", "Ratio contain relative", "Mean baseline approximation ratio" ])
        writer.writerow([ mean_rational_size.item(), ratio_no_distractor.item(), ratio_contain_relative.item(), mean_baseline_approximation_ratio.item() ])
