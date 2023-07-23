
import argparse
import json
import logging
import os

import torch

from transformers import AutoTokenizer
from natsort import natsorted


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", 
                        type=str,
                        default="data/analogies",
                        help="") # TODO
    parser.add_argument("--input-dir", 
                        type=str,
                        default="../sequential-rationales/huggingface/rationalization_results/analogies/greedy",
                        help="Dir to greedy results") # TODO
    parser.add_argument("--output-path", 
                        type=str,
                        default="rationalization_results/analogies-greedy-lengths.json",
                        help="") # TODO
    parser.add_argument("--tokenizer", 
                        type=str,
                        default="gpt2-medium",
                        help="") # TODO
    args = parser.parse_args()

    data_dir = args.data_dir
    input_dir = args.input_dir
    output_path = args.output_path
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    dirpath, dirnames, filenames = next(os.walk(data_dir))
    # filenames.sort()
    filenames = natsorted(filenames)

    rational_size_map = {}

    for filename in filenames:

        with open(os.path.join(dirpath, filename)) as data_f:
            data = json.load(data_f)
        
        analogy_idx = data["comments"]["analogy_idx"]
        pair_idx = data["comments"]["pair_idx"]

        result_old_path = os.path.join(input_dir, f"{analogy_idx}_{pair_idx}.json")
        if not os.path.exists(result_old_path):
            logging.warning(f"[Warning] {result_old_path} not found. Skipping {filename}")
            continue

        with open(result_old_path) as result_old_f:
            result_old = json.load(result_old_f)

        all_rationals = torch.tensor(result_old["all_rationales"][0])
        rational_size = all_rationals.shape[0]

        rational_size_map[filename] = rational_size
    
        logging.info(filename)

    compact = False
    indent = None if compact else 4
    json_str = json.dumps(rational_size_map, indent=indent)

    with open(output_path, 'w') as f_output:
        f_output.write(json_str)
    
    logging.info("done")
