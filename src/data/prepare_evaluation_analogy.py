import argparse
import json
import os

import torch
from data_utils import create_analogy_templates, preprocess_analogies
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--analogies-file", type=str, default="data/analogies.txt", help=""
)  # TODO
parser.add_argument("--output-dir", type=str, default="data/analogies", help="")  # TODO
parser.add_argument("--compact-output", type=bool, default=True, help="")  # TODO
parser.add_argument(
    "--schema-uri", type=str, default="../../docs/analogy.schema.json", help=""
)  # TODO

args = parser.parse_args()
analogies_file = args.analogies_file
output_dir = args.output_dir
compact_output = args.compact_output
schema_uri = args.schema_uri

# Load analogies

with open(analogies_file) as f:
    analogies = f.readlines()
analogies = [line.rstrip("\n") for line in analogies]

# Load model

tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
model.cuda()
model.eval()

# Build analogies index

all_analogies = preprocess_analogies(analogies, tokenizer)
all_analogies = create_analogy_templates(all_analogies)

# Build data

data_id = 0
for analogy_idx, (analogy_label, analogy_config) in enumerate(all_analogies.items()):
    analogy_config = all_analogies[analogy_label]
    template = analogy_config["template"]

    # map tags to target/relative
    target_tag = "a" if template.index("[A]") > template.index("[B]") else "b"
    relative_tag = "a" if target_tag == "b" else "b"

    for pair_idx in range(len(analogy_config["a"])):
        word_a = analogy_config["a"][pair_idx]
        word_b = analogy_config["b"][pair_idx]

        full_sentence = template.replace("[A]", word_a).replace("[B]", word_b)
        full_ids = tokenizer(full_sentence, return_tensors="pt")["input_ids"].cuda()

        model_prediction_full = tokenizer.decode(
            model.generate(
                full_ids[:, :-1], max_length=full_ids.shape[1], do_sample=False
            )[0]
        )

        distractor_start_id = tokenizer.encode(" (")[0]
        distractor_end_id = tokenizer.encode(").")[0]

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
            target_word_id = tokenizer.encode(" " + target_word)[0]
            target_pos = torch.nonzero(full_ids[0] == target_word_id)[0, 0]
            relative_word_id = tokenizer.encode(" " + relative_word)[0]
            relative_pos = torch.nonzero(full_ids[0] == relative_word_id)[0, 0]

            full_sentence_list = [
                tokenizer.decode([token_id]) for token_id in full_ids[0]
            ]
            full_ids_list = [v.item() for v in full_ids[0]]
            data = {
                "$schema": schema_uri,
                "id": data_id,
                "input-text": full_sentence_list,
                "input-tokens": full_ids_list,
                "target": target_pos.item(),
                "relative": relative_pos.item(),
                "distractor": {
                    "start": distractor_start_pos.item(),
                    "end": distractor_end_pos.item(),
                },
                "comments": {
                    "tokenizer": "gpt2-medium",
                    "model": "gpt2-medium",
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

            data_id += 1
            print(data_id)
