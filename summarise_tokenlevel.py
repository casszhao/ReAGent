import pandas as pd
import seaborn as sns
import os 
import matplotlib.pyplot as plt
import numpy as np


model_name="gpt2"
hyper="top3_replace0.1_max5000_batch5"
flex = False

def get_one_line_for_one_FA(model_name, FA_name,ratio_list):
    eva_output_dir=f"evaluation_results/analogies/{model_name}_{FA_name}"
    if FA_name == 'ours': eva_output_dir=f"evaluation_results/analogies/{model_name}_{FA_name}/{hyper}"
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
        print('=========> ', ratio)
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
        
        elif ratio == 1: # mean and fix len
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

def minus_and_save(suff_df, random_suff_df, save_name):
    final_suff_df = suff_df.copy()
    for col in suff_df.columns:
        for row in suff_df.index:
            if isinstance(suff_df.at[row, col], float) and isinstance(random_suff_df.at[row, col], float):
                final_suff_df.at[row, col] = suff_df.at[row, col] - random_suff_df.at[row, col]

    print(' =======>   final divided results =======')
    print(final_suff_df)
    os.makedirs(f'evaluation_results/summary/{model_name}/', exist_ok=True)
    final_suff_df.to_csv(f'evaluation_results/summary/{model_name}/{save_name}_{hyper}_minus.csv')
    return final_suff_df
    


if flex == True:
    ratio_list = [0.05, 0.1, 0.2, 0.3, 0.0, 1.0] # 0 here for flexible len from greedy search and 1 for soft
    columns_suff_name = ['FAs','5% Suff', '10% Suff', '20% Suff', '30% Suff', 'Mean Suff', 'FlexLen Suff', 'Soft Suff']
    columns_comp_name = ['FAs','5% Comp', '10% Comp', '20% Comp', '30% Comp', 'Mean Comp', 'FlexLen Comp', 'Soft Comp']
else: 
    ratio_list = [0.05, 0.1, 0.2, 0.3, 1.0]
    columns_suff_name = ['FAs','5% Suff', '10% Suff', '20% Suff', '30% Suff', 'Mean Suff', 'Soft Suff']
    columns_comp_name = ['FAs','5% Comp', '10% Comp', '20% Comp', '30% Comp', 'Mean Comp', 'Soft Comp']


def get_one_model(model_name, ratio_list):
    rollout_suff, rollout_comp, random_rollout_suff, random_rollout_comp = get_one_line_for_one_FA(model_name, "attention_rollout", ratio_list)
    last_suff, last_comp, random_last_suff, random_last_comp = get_one_line_for_one_FA(model_name, "attention_last", ratio_list)
    all_suff, all_comp, random_all_suff, random_all_comp = get_one_line_for_one_FA(model_name, "attention", ratio_list)

    #norms_suff, norms_comp, random_norms_suff, random_norms_comp = get_one_line_for_one_FA(model_name, "norm", ratio_list)
    signed_suff, signed_comp, random_signed_suff, random_signed_comp = get_one_line_for_one_FA(model_name, "input_x_gradient", ratio_list)
    integrated_suff, integrated_comp, random_integrated_suff, random_integrated_comp = get_one_line_for_one_FA(model_name, "integrated_gradients", ratio_list)
    shap_suff, shap_comp, random_shap_suff, random_shap_comp = get_one_line_for_one_FA(model_name, "gradient_shap", ratio_list)


    ours_suff, ours_comp, ours_random_all_suff, ours_random_all_comp = get_one_line_for_one_FA(model_name, "ours", ratio_list)

    suff_df = pd.DataFrame([signed_suff, integrated_suff, shap_suff, rollout_suff, last_suff, all_suff, ours_suff], columns=columns_suff_name)
    comp_df = pd.DataFrame([signed_comp, integrated_comp, shap_comp, rollout_comp, last_comp, all_comp, ours_comp], columns=columns_comp_name)

    random_suff_df = pd.DataFrame([random_signed_suff, random_integrated_suff, random_shap_suff, random_rollout_suff, random_last_suff, random_all_suff, ours_random_all_suff], columns=columns_suff_name)
    random_comp_df = pd.DataFrame([random_signed_comp, random_integrated_comp, random_shap_comp, random_rollout_comp, random_last_comp, random_all_comp, ours_random_all_comp], columns=columns_comp_name)
    #print(f"random_comp_df ==>> {random_comp_df}")


    print(' ========== SUFF =========')
    print(suff_df)
    print(' ========== RANDOM SUFF =========')
    print(random_suff_df)



    final_suff_df = minus_and_save(suff_df, random_suff_df, 'suff')
    final_comp_df = minus_and_save(comp_df, random_comp_df, 'comp')


    stacked_df = pd.concat([final_suff_df, final_comp_df])
    stacked_df.to_csv(f'evaluation_results/summary/{model_name}/{hyper}_minus.csv')

    print(' Done')
    return stacked_df



model_name_dict = { 'gpt2':'GPT2 354M', 'gpt2_xl': 'GPT2 1.5B', 'gpt6b': 'GPT-J 6B', \
                    'OPT350M': 'OPT 350M', 'OPT1B':'OPT 1.3B', 'OPT6B':'OPT 6.7B', \
                    'GradxEmb':'Grad x Emb', }


all_results = []

for model_name in ['gpt2', 'gpt2_xl', 'OPT1B', 'OPT6B']: #'gpt6b', 'OPT350M', 
    temp = get_one_model(model_name, ratio_list)
    temp['Model'] = model_name
    all_results.append(temp)


df = pd.concat(all_results)

df.replace(model_name_dict,inplace=True)

suff = df[["FAs","Soft Suff","Model"]]
comp = df[["FAs","Soft Comp","Model"]]


plt.figure(figsize=(22, 22))
fig, axs = plt.subplots(nrows=3, ncols=1, sharex=False, ) #squeeze=True,
#fig.title('Wikitext sentence-level faithfulness')

plt.subplot(3,1,3)  # row colum
axs[2].set_visible(False)

plt.subplot(3,1,1)  # row colum
sns.barplot(x="Model", y="Soft Comp", hue="FAs", data=comp, errorbar=None, width= 0.6,
            order=['OPT 350M','OPT 1.3B', 'OPT 6.7B','GPT2 354M', 'GPT2 1.5B']) #, 'GPT-J 6B'
plt.xlabel('Models', fontweight='bold')

plt.subplot(3,1,2)  # row colum
sns.barplot(x="Model", y="Soft Suff", hue="FAs", data=suff, errorbar=None,  width= 0.6,
            order=['OPT 350M','OPT 1.3B', 'OPT 6.7B','GPT2 354M', 'GPT2 1.5B']) # , height=8
plt.xlabel('Models', fontweight='bold')



handles, labels = axs[0].get_legend_handles_labels()

fig.legend(handles[:8], labels[:8], ncol=4, loc='center', bbox_to_anchor=(0.5, 0.21), fontsize=9) # 00 0.4 middle 0.8 top
# fig.legend(nrow=1, loc='lower right', bbox_to_anchor=(1.19, 0.1)) #
plt.legend()
axs[0].get_legend().remove()
axs[1].get_legend().remove()
# Add xticks on the middle of the group bars

fig.suptitle('Token-level Faithfulness', fontsize=15)
fig.tight_layout() 
plt.show()

plt.savefig(f"./evaluation_results/summary/analogies/token.png",bbox_inches='tight')
