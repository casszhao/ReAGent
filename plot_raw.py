import argparse
import csv
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()

parser.add_argument("--tokenizer-name", 
                    type=str,
                    default="facebook/opt-1.3b")

parser.add_argument("--source-data-path", 
                    type=str,
                    default="data/tellmewhy2.txt")

parser.add_argument("--result-raw-dir", 
                    type=str,
                    default="evaluation_results/")

parser.add_argument("--output-dir", 
                    type=str,
                    default="plots")

args = parser.parse_args()

# tokenizer_name = 'facebook/opt-1.3b'
# source_data_path = 'tellmewhy2.txt'
# result_raw_dir = './'
# output_dir = ''

tokenizer_name = args.tokenizer_name
source_data_path = args.source_data_path
result_raw_dir = args.result_raw_dir
output_dir = args.output_dir

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir='cache')

with open(source_data_path, "r") as in_f:
    input_text_list = in_f.read().splitlines()

for i, input_text in enumerate(input_text_list):

    try:

        tokens_text = [ tokenizer.decode([token_id]) for token_id in tokenizer.encode(input_text) ]

        with open(os.path.join(result_raw_dir, f'{i}_raw.json')) as raw_f:
            raw_dict = json.loads(raw_f.read())

        index_dict = { v['target_pos'] : (k, v) for k, v in raw_dict.items() }

        current_pos = len(tokens_text)

        with open(os.path.join(result_raw_dir, f'{i}_details.csv'), newline='') as csvfile:
            details_reader = csv.DictReader(csvfile)
            for row in details_reader:
                target_pos = int(row['target_pos']) - 1
                target_token = row['target_token']

                if (target_pos != current_pos):
                    print(f'Sample {i}, Pos {target_pos}. discontinuous record, terminated.')
                    break

                if (target_pos in index_dict):

                    result_item_key, result_item_value = index_dict[target_pos]

                    importance_distribution = result_item_value['importance_distribution']       

                    print(f'Sample {i}, Pos {target_pos}, Token {result_item_key}')

                    # plot
                    import numpy as np
                    import seaborn

                    importance_distribution_np = np.array([importance_distribution])
                    s = seaborn.heatmap(
                        importance_distribution_np, 
                        xticklabels=tokens_text, 
                        yticklabels=[result_item_key],
                        annot=True, 
                        square=True)
                    fig = s.get_figure()
                    fig.tight_layout()
                    fig.savefig(os.path.join(output_dir, f'{i}_{target_pos}.png'))
                    fig.clf()

                else:
                    print(f'Sample {i}, Pos {target_pos}. attribution record not found. skipped')

                current_pos += 1
                tokens_text.append(target_token)
                    
    except FileNotFoundError:
        print(f'Sample {i}, Record not found. skipped')