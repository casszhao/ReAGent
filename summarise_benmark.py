import pandas as pd

model_name="gpt2"
hyper="top5_replace0.3_max5000_batch8"

def get_one_line_for_one_FA(model_name, FA_name, task_name):
    print(' ====> ', FA_name)
    eva_output_dir=f"evaluation_results/benchmark/{model_name}_{FA_name}/{task_name}"
    #if FA_name == 'ours': eva_output_dir=f"evaluation_results/analogies/{model_name}_{FA_name}/{hyper}"
    #print(f"==>> eva_output_dir: {eva_output_dir}")


    # suff_list = [FA_name.replace('_', ' ').title()]
    # comp_list = [FA_name.replace('_', ' ').title()]
    suff_mean = 0
    comp_mean = 0

    # random_suff_list = [FA_name.replace('_', ' ').title()]
    # random_comp_list = [FA_name.replace('_', ' ').title()]
    random_suff_mean = 0
    random_comp_mean = 0


    
    #diff_ratio_len = int(len(ratio_list)-1)
    for file_id in range(0, 5):
        faithful_results = pd.read_csv(eva_output_dir+f'/{file_id}_mean.csv') # one data

        suff_mean += faithful_results['final suff'][0]
        comp_mean += faithful_results['final comp'][0]

    print(suff_mean/5, comp_mean/5)
    return suff_mean/5, comp_mean/5

signed_suff, signed_comp = get_one_line_for_one_FA(model_name, "input_x_gradient", 'wikitext')
integrated_suff, integrated_comp = get_one_line_for_one_FA(model_name, "integrated_gradients", 'wikitext')
integrated_suff, integrated_comp = get_one_line_for_one_FA(model_name, "gradient_shap", 'wikitext')
integrated_suff, integrated_comp = get_one_line_for_one_FA(model_name, "attention", 'wikitext')
ours_suff, ours_comp = get_one_line_for_one_FA(model_name, "ours", 'wikitext')

quit()

suff_df = pd.DataFrame([norms_suff, signed_suff, integrated_suff, rollout_suff, last_suff, all_suff, ours_suff], columns=['Method','5% Suff', '10% Suff', '20% Suff', '30% Suff', 'Mean Suff', 'FlexLen Suff', 'Soft Suff'])
comp_df = pd.DataFrame([norms_comp, signed_comp, integrated_comp, rollout_comp, last_comp, all_comp, ours_comp], columns=['Method','5% Comp', '10% Comp', '20% Comp', '30% Comp', 'Mean Comp', 'FlexLen Comp', 'Soft Comp'])
#print(f"comp_df ==>> {comp_df}")
# suff_df = pd.DataFrame([rollout_suff, last_suff, all_suff, ours_suff], columns=['Method','fix len Suff'])
# comp_df = pd.DataFrame([rollout_comp, last_comp, all_comp, ours_comp], columns=['Method','fix len Comp'])

random_suff_df = pd.DataFrame([random_norms_suff, random_signed_suff, random_integrated_suff, random_rollout_suff, random_last_suff, random_all_suff, ours_random_all_suff], columns=['Method','5% Suff', '10% Suff', '20% Suff', '30% Suff', 'Mean Suff', 'FlexLen Suff', 'Soft Suff'])
random_comp_df = pd.DataFrame([random_norms_comp, random_signed_comp, random_integrated_comp, random_rollout_comp, random_last_comp, random_all_comp, ours_random_all_comp], columns=['Method','5% Comp', '10% Comp', '20% Comp', '30% Comp', 'Mean Comp', 'FlexLen Comp', 'Soft Comp'])
#print(f"random_comp_df ==>> {random_comp_df}")


print(' ========== SUFF =========')
print(suff_df)
print(' ========== RANDOM SUFF =========')
print(random_suff_df)
# print(' ')
# print(' ')
# print(' ========== Comp =========')
# print(comp_df)
# print(' ========== RANDOM Comp =========')
# print(random_comp_df)
# print(' ')
# print(' ')


def div_and_save(suff_df, random_suff_df, save_name):
    final_suff_df = suff_df.copy()
    for col in suff_df.columns:
        for row in suff_df.index:
            if isinstance(suff_df.at[row, col], float) and isinstance(random_suff_df.at[row, col], float):
                final_suff_df.at[row, col] = suff_df.at[row, col] / random_suff_df.at[row, col]

    print(' =======>   final divided results =======')
    print(final_suff_df)
    final_suff_df.to_csv(f'evaluation_results/summary/{model_name}/{save_name}_{hyper}.csv')
    return final_suff_df
    
final_suff_df = div_and_save(suff_df, random_suff_df, 'suff')
final_comp_df = div_and_save(comp_df, random_comp_df, 'comp')


stacked_df = pd.concat([final_suff_df, final_comp_df])
stacked_df.to_csv(f'evaluation_results/summary/{model_name}/{hyper}.csv')

print(' Done')