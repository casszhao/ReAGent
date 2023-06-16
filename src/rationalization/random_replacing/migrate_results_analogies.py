
import argparse
import json
import logging
import os

import torch

from utils.serializing import serialize_rational
from transformers import AutoTokenizer


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", 
                        type=str,
                        default="data/analogies",
                        help="") # TODO
    parser.add_argument("--input-dir", 
                        type=str,
                        default="rationalization_results/analogies-old/last_attention",
                        help="") # TODO
    parser.add_argument("--output-dir", 
                        type=str,
                        default="rationalization_results/analogies/gpt2-medium.last_attention",
                        help="") # TODO
    parser.add_argument("--tokenizer", 
                        type=str,
                        default="gpt2-medium",
                        help="") # TODO
    args = parser.parse_args()

    data_dir = args.data_dir
    input_dir = args.input_dir
    output_dir = args.output_dir
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    dirpath, dirnames, filenames = next(os.walk(data_dir))
    filenames.sort()

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

        output_filename = os.path.join(output_dir, filename)
        serialize_rational(
            output_filename,
            data["id"], 
            torch.tensor(result_old["input_ids"][:-1]), 
            torch.tensor(result_old["input_ids"][-1]), 
            torch.tensor(result_old["all_rationales"][-1]), 
            tokenizer, 
            None,
            compact=False,
            comments= {
                "created-by": os.path.basename(__file__),
                "args" : args.__dict__
            },
            # trace_rationalizer=rationalizer # Enable trace logs
        )

        logging.info(filename)