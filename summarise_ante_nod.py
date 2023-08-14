import pandas as pd

model_name="gpt2"
hyper="top5_replace0.3_max5000_batch8"

def get_one_line_for_one_FA(model_name, FA_name,len_list):
    eva_output_dir=f"evaluation_results/analogies/{model_name}_{FA_name}"
    if FA_name == 'ours': eva_output_dir=f"evaluation_results/analogies/{model_name}_{FA_name}/{hyper}"

    column_name = ['Methods']
    fa_list = [FA_name.replace('_', ' ').title()]

    for len in len_list:
        faithful_results = pd.read_csv(eva_output_dir+f'/{len}_ante_nod.csv')

        column_name.append(f'Ante ({len})')
        column_name.append(f'No D ({len})')
        
        fa_list.append(faithful_results['Ratio contain relative'][0])
        fa_list.append(faithful_results['Ratio no distractor'][0])

    return fa_list, column_name




ratio_list = [3,5,7,10] # 0 here for flexible len from greedy search and 1 for soft
rollout, column_name = get_one_line_for_one_FA(model_name, "rollout_attention", ratio_list)
last, column_name = get_one_line_for_one_FA(model_name, "last_attention", ratio_list)
all, column_name = get_one_line_for_one_FA(model_name, "all_attention", ratio_list)

norms, column_name = get_one_line_for_one_FA(model_name, "norm", ratio_list)
signed, column_name = get_one_line_for_one_FA(model_name, "signed", ratio_list)
integrated, column_name = get_one_line_for_one_FA(model_name, "inseq_ig", ratio_list)

ours, column_name = get_one_line_for_one_FA(model_name, "ours", ratio_list)

df = pd.DataFrame([norms, signed, integrated, rollout, last, all, ours], columns=column_name)
import os
os.makedirs(f'evaluation_results/summary/{model_name}_ante_nod/', exist_ok=True)
df.to_csv(f'evaluation_results/summary/{model_name}_ante_nod/{hyper}.csv')

print(' Done')