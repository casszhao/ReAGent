import argparse
import json
import logging
import os

import torch
from data_utils import create_analogy_templates, preprocess_analogies
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument( "--analogies-file", type=str, default="data/analogies.txt", help="")  # TODO
parser.add_argument("--output-dir", type=str, default="data/analogies", help="")  # TODO
parser.add_argument("--compact-output", type=bool, default=True, help="")  # TODO
parser.add_argument("--schema-uri", type=str, default="../../docs/analogy.schema.json", help="")  # TODO
parser.add_argument("--device", type=str, default="cuda", help="")  # TODO

parser.add_argument("--model", type=str, default="gpt2-medium", help="") # TODO # gpt2-medium gpt2-large 
parser.add_argument("--cache_dir", type=str, default=None, help="store models")
args = parser.parse_args()

analogies_file = args.analogies_file
output_dir = args.output_dir
compact_output = args.compact_output
schema_uri = args.schema_uri
device = args.device

# Load analogies

with open(analogies_file) as f:
    analogies = f.readlines()
analogies = [line.rstrip("\n") for line in analogies]
print("00".center(50, "-"))
print(f"==>> analogies: {analogies}")

# Load model

tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir=args.cache_dir)
model.to(device)

with torch.no_grad():

    # Build analogies index

    all_analogies = preprocess_analogies(analogies, tokenizer)
    #print(f"1111111==>> all_analogies: {all_analogies}")
    all_analogies = create_analogy_templates(all_analogies)
    print(' all_analogies', all_analogies.keys)
    print(' all_analogies', all_analogies['currency'])
    print(' ========= ')
    #print(' all_analogies', all_analogies)

    # Build data

    data_id = 0
    for analogy_idx, (analogy_label, analogy_config) in enumerate(all_analogies.items()):
        print(analogy_idx)
        print(analogy_label)
        print(analogy_config)
        analogy_config = all_analogies[analogy_label]
        
        template = analogy_config["template"]
        print(f"==>> template: {template}")

        # map tags to target/relative
        target_tag = "a" if template.index("[A]") > template.index("[B]") else "b"
        print(f"==>> target_tag: {target_tag}")
        relative_tag = "a" if target_tag == "b" else "b"
        print(f"==>> relative_tag: {relative_tag}")

        print(f"==>> analogy_config: {analogy_config}")

        for pair_idx in range(len(analogy_config["a"])):
            print("".center(50, "-"))
            print(pair_idx)
            word_a = analogy_config["a"][pair_idx]
            word_b = analogy_config["b"][pair_idx]

            full_sentence = template.replace("[A]", word_a).replace("[B]", word_b)
            print(full_sentence)
            full_ids = tokenizer(full_sentence, return_tensors="pt")["input_ids"].cuda()

            model_prediction_full = tokenizer.decode(
                model.generate(
                    full_ids[:, :-1], max_length=full_ids.shape[1], do_sample=False
                )[0]
            )
            

            if "llama" in str(args.model): 
                print('Confirmed ===> model is llama')
                distractor_start_id = 353
                distractor_end_id = 621
                
            else: 
                distractor_start_id = tokenizer.encode(" (")[-1]
                distractor_end_id = tokenizer.encode(").")[-1]
            
            print(full_ids[0])
            distractor_start_pos = torch.nonzero(full_ids[0] == distractor_start_id)[0, 0]
            distractor_end_pos = torch.nonzero(full_ids[0] == distractor_end_id)[0, 0]

            distractor_removed = torch.cat(
                [full_ids[:, :distractor_start_pos], full_ids[:, distractor_end_pos + 1 :]],
                1,
            )
            model_prediction_no_distractor = tokenizer.decode(
                model.generate(
                    distractor_removed[:, :-1],
                    max_length=distractor_removed.shape[1],
                    do_sample=False,
                )[0]
            )

            # Only rationalize analogies the model predicts correctly, both with
            # and without the distractor
            relative_word = analogy_config[relative_tag][pair_idx]
            target_word = analogy_config[target_tag][pair_idx]
            if (
                model_prediction_full.split(" ")[-1] == target_word
                and model_prediction_no_distractor.split(" ")[-1] == target_word
            ):
                print("".center(50, "-"))
                print("".center(50, "-"))
                print("".center(50, "-"))
                
                target_word_id = tokenizer.encode(" " + target_word)[-1]
                print(f"==>> relative_word_id: ", tokenizer.encode(" " + relative_word))  # for OPT2: ==>> relative_word_id:  [2, 173]
                relative_word_id = tokenizer.encode(" " + relative_word)[-1]
                
                target_pos = torch.nonzero(full_ids[0] == target_word_id)[0, 0]
                relative_pos = torch.nonzero(full_ids[0] == relative_word_id)[0, 0]

                full_sentence_list = [
                    tokenizer.decode([token_id]) for token_id in full_ids[0]
                ]
                full_ids_list = [v.item() for v in full_ids[0]]
                data = {
                    "$schema": schema_uri,
                    "id": data_id,
                    "text": full_sentence_list,
                    "tokens": full_ids_list,
                    "target": target_pos.item(),
                    "relative": relative_pos.item(),
                    "distractor": {
                        "start": distractor_start_pos.item(),
                        "end": distractor_end_pos.item(),
                    },
                    "comments": {
                        "tokenizer": str(args.model),
                        "model": str(args.model),
                        "analogy_idx": analogy_idx,
                        "pair_idx": pair_idx,
                    },
                }

                # export file
                indent = None if compact_output else 4
                json_str = json.dumps(data, indent=indent)

                filename = os.path.join(output_dir, f"{data_id}.json")
                with open(filename, "w") as f_output:
                    f_output.write(json_str)
                    print(filename)

                data_id += 1
                logging.info(f"done {data_id}")
