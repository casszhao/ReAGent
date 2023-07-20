import pandas as pd

model_name="gpt2-medium"
FA_name="last_attention"

importance_results=f"rationalization_results/analogies/{model_name}_{FA_name}"
eva_output_dir=f"evaluation_results/analogies/{model_name}_{FA_name}"

faithful_results = pd.read_csv(eva_output_dir+'/mean.csv')
print(f"==>> faithful_results: {faithful_results}")
final_list=[]

metrics_list = ['suff', 'soft_suff', 'comp', 'soft_comp']

for metrics in metrics_list:
    final=faithful_results[f'{metrics}']/faithful_results[f'random_{metrics}']
    final_list.append(final)

    

print(f"==>> metrics: {metrics_list}")
print(f"==>> final: {final_list}")