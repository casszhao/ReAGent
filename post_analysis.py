import pandas as pd

model_name="gpt2"
hyper="/top5_replace0.3_max3000"

def get_one_line_for_one_FA(model_name, FA_name,ratio_list):
    eva_output_dir=f"evaluation_results/analogies/{model_name}_{FA_name}"
    print(f"==>> eva_output_dir: {eva_output_dir}")


    suff_list = [FA_name.replace('_', ' ').title()]
    comp_list = [FA_name.replace('_', ' ').title()]
    suff_mean = 0
    comp_mean = 0

    random_suff_list = [FA_name.replace('_', ' ').title()]
    random_comp_list = [FA_name.replace('_', ' ').title()]
    random_suff_mean = 0
    random_comp_mean = 0


    
    diff_ratio_len = int(len(ratio_list)-1)
    for ratio in ratio_list:
        faithful_results = pd.read_csv(eva_output_dir+f'/mean_{ratio}.csv')

        if 0 < ratio < 1: # 0.05 - 0.3
            suff_list.append(faithful_results['suff'][0])
            comp_list.append(faithful_results['comp'][0])
            suff_mean += faithful_results['suff'][0]
            comp_mean += faithful_results['comp'][0]

            random_suff_list.append(faithful_results['random_suff'][0])
            random_comp_list.append(faithful_results['random_comp'][0])
            random_suff_mean += faithful_results['random_suff'][0]
            random_comp_mean += faithful_results['random_comp'][0]
        
        elif ratio == 0: # mean and fix len
            suff_list.append(suff_mean/diff_ratio_len)
            suff_list.append(faithful_results['suff'][0])

            comp_list.append(comp_mean/diff_ratio_len)
            comp_list.append(faithful_results['comp'][0])

            random_suff_list.append(random_suff_mean/diff_ratio_len)
            random_suff_list.append(faithful_results['random_suff'][0])

            random_comp_list.append(random_comp_mean/diff_ratio_len)
            random_comp_list.append(faithful_results['random_comp'][0])
        
        else: # soft
            suff_list.append(faithful_results['suff'][0])
            comp_list.append(faithful_results['comp'][0])
            random_suff_list.append(faithful_results['random_suff'][0])
            random_comp_list.append(faithful_results['random_comp'][0])


    return suff_list, comp_list, random_suff_list, random_comp_list


ratio_list = [0.05, 0.1, 0.2, 0.3, 0.0, 1.0] # 0 here for flexible len from greedy search and 1 for soft
rollout_suff, rollout_comp, random_rollout_suff, random_rollout_comp = get_one_line_for_one_FA(model_name, "rollout_attention", ratio_list)
last_suff, last_comp, random_last_suff, random_last_comp = get_one_line_for_one_FA(model_name, "last_attention", ratio_list)
all_suff, all_comp, random_all_suff, random_all_comp = get_one_line_for_one_FA(model_name, "all_attention", ratio_list)

norms_suff, norms_comp, random_norms_suff, random_norms_comp = get_one_line_for_one_FA(model_name, "norm", ratio_list)
signed_suff, signed_comp, random_signed_suff, random_signed_comp = get_one_line_for_one_FA(model_name, "signed", ratio_list)
integrated_suff, integrated_comp, random_integrated_suff, random_integrated_comp = get_one_line_for_one_FA(model_name, "integrated", ratio_list)
ours_suff, ours_comp, ours_random_all_suff, ours_random_all_comp = get_one_line_for_one_FA(model_name+hyper, "ours", ratio_list)

suff_df = pd.DataFrame([norms_suff, signed_suff, integrated_suff, rollout_suff, last_suff, all_suff, ours_suff], columns=['Method','5% Suff', '10% Suff', '20% Suff', '30% Suff', 'Mean Suff', 'FlexLen Suff', 'Soft Suff'])
comp_df = pd.DataFrame([norms_comp, signed_comp, integrated_comp, rollout_comp, last_comp, all_comp, ours_comp], columns=['Method','5% Comp', '10% Comp', '20% Comp', '30% Comp', 'Mean Comp', 'FlexLen Comp', 'Soft Comp'])
# suff_df = pd.DataFrame([rollout_suff, last_suff, all_suff, ours_suff], columns=['Method','fix len Suff'])
# comp_df = pd.DataFrame([rollout_comp, last_comp, all_comp, ours_comp], columns=['Method','fix len Comp'])

random_suff_df = pd.DataFrame([random_norms_suff, random_signed_suff, random_integrated_suff, random_rollout_suff, random_last_suff, random_all_suff, ours_random_all_suff], columns=['Method','5% Suff', '10% Suff', '20% Suff', '30% Suff', 'Mean Suff', 'FlexLen Suff', 'Soft Suff'])
random_comp_df = pd.DataFrame([random_norms_comp, random_signed_comp, random_integrated_comp, random_rollout_comp, random_last_comp, random_all_comp, ours_random_all_comp], columns=['Method','5% Comp', '10% Comp', '20% Comp', '30% Comp', 'Mean Comp', 'FlexLen Comp', 'Soft Comp'])

print(suff_df)
print(' ')
print(random_suff_df)
print(' ')
print(' ')
print(comp_df)
print(' ')
print(random_comp_df)
print(' ')
print(' ')


def div_and_save(suff_df, random_suff_df, save_name):
    final_suff_df = suff_df.copy()
    for col in suff_df.columns:
        for row in suff_df.index:
            if isinstance(suff_df.at[row, col], float) and isinstance(random_suff_df.at[row, col], float):
                final_suff_df.at[row, col] = suff_df.at[row, col] / random_suff_df.at[row, col]

    print(final_suff_df)
    final_suff_df.to_csv(f'evaluation_results/summary/faith_summary_{model_name}_{save_name}.csv')
    return final_suff_df
    
final_suff_df = div_and_save(suff_df, random_suff_df, 'suff')
final_comp_df = div_and_save(comp_df, random_comp_df, 'comp')

stacked_df = pd.concat([final_suff_df, final_comp_df])
stacked_df.to_csv(f'evaluation_results/summary/faith_summary_{model_name}.csv')