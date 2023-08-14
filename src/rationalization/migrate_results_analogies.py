
import argparse
import json
import logging
import os

import torch

from rationalizer.utils.serializing import serialize_rational
from transformers import AutoTokenizer
from natsort import natsorted

import logging


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", 
                        type=str,
                        default="data/analogies/gpt2/",
                        help="") # TODO
    parser.add_argument("--input_dir", 
                        type=str,
                        default="/mnt/parscratch/users/cass/seq_rationales/rationalization_results/analogies/gpt2_exhaustive/",
                        #default="../../sequential-rationales/huggingface/rationalization_results/analogies/gpt2_exhaustive/",
                        help="") # TODO
    parser.add_argument("--output_dir", 
                        type=str,
                        default="rationalization_results/analogies/gpt2_exhaustive",
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
    # filenames.sort()
    filenames = natsorted(filenames)

    for filename in filenames:

        with open(os.path.join(dirpath, filename)) as data_f:
            data = json.load(data_f)
        
        analogy_idx = data["comments"]["analogy_idx"]
        pair_idx = data["comments"]["pair_idx"]

        

        result_old_path = os.path.join(input_dir, f"{analogy_idx}_{pair_idx}.json")
        #logging.debug(' about to convert ', result_old_path)
        if not os.path.exists(result_old_path):
            logging.warning(f"[Warning] {result_old_path} not found. Skipping {filename}")
            continue

        with open(result_old_path) as result_old_f:
            result_old = json.load(result_old_f)

        output_filename = os.path.join(output_dir, filename)

        all_rationals = torch.tensor(result_old["all_rationales"][0])
        seq_length = len(result_old["input_ids"]) - 1
        importance_mask = torch.zeros([seq_length]).scatter(-1, all_rationals, 1)
        importance_score = importance_mask / torch.sum(importance_mask)

        serialize_rational(
            output_filename,
            data["id"], 
            torch.tensor(result_old["input_ids"][:-1]), 
            torch.tensor(result_old["input_ids"][-1]), 
            torch.tensor(result_old["all_rationales"][-1]), 
            tokenizer, 
            importance_score,
            compact=False,
            comments= {
                "created-by": os.path.basename(__file__),
                "args" : args.__dict__,
                "note": "pseudo importance score applied"
            },
            # trace_rationalizer=rationalizer # Enable trace logs
        )

        logging.info(filename)
