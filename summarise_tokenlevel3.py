

import pandas as pd
import seaborn as sns
import os 
import matplotlib.pyplot as plt
import numpy as np




model_name="gpt2"
hyper="top3_replace0.1_max5000_batch5"
#hyper="top3_replace0.3_max3000_batch5"
hyper2="top3_replace0.3_max3000_batch10"
# "/top3_replace0.3_max3000_batch5"
flex = False

def get_one_line_for_one_FA(model_name, FA_name):

    suff_list = [FA_name.replace('_', ' ').title()]
    comp_list = [FA_name.replace('_', ' ').title()]

    random_suff_list = [FA_name.replace('_', ' ').title()]
    random_comp_list = [FA_name.replace('_', ' ').title()]

    eva_output_dir=f"evaluation_results/analogies/{model_name}_{FA_name}"

    if FA_name == 'ours': 
        try: 
            eva_output_dir=f"evaluation_results/analogies/{model_name}_{FA_name}/{hyper}"
            faithful_results = pd.read_csv(eva_output_dir+'/mean_1.0.csv')
        except:
            eva_output_dir=f"evaluation_results/analogies/{model_name}_{FA_name}/{hyper2}"
            faithful_results = pd.read_csv(eva_output_dir+'/mean_1.0.csv')
    else:
        faithful_results = pd.read_csv(eva_output_dir+'/mean_1.0.csv')

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
                final_suff_df.at[row, col] = suff_df.at[row, col] / random_suff_df.at[row, col]

    print(' =======>   final divided results =======')
    print(final_suff_df)
    os.makedirs(f'evaluation_results/summary/{model_name}/', exist_ok=True)
    final_suff_df.to_csv(f'evaluation_results/summary/{model_name}/{save_name}_{hyper}_minus.csv')
    return final_suff_df
    

def get_one_model(model_name):
    rollout_suff, rollout_comp, random_rollout_suff, random_rollout_comp = get_one_line_for_one_FA(model_name, "attention_rollout")
    last_suff, last_comp, random_last_suff, random_last_comp = get_one_line_for_one_FA(model_name, "attention_last")
    all_suff, all_comp, random_all_suff, random_all_comp = get_one_line_for_one_FA(model_name, "attention")

    signed_suff, signed_comp, random_signed_suff, random_signed_comp = get_one_line_for_one_FA(model_name, "input_x_gradient")
    integrated_suff, integrated_comp, random_integrated_suff, random_integrated_comp = get_one_line_for_one_FA(model_name, "integrated_gradients")
    shap_suff, shap_comp, random_shap_suff, random_shap_comp = get_one_line_for_one_FA(model_name, "gradient_shap")

    ours_suff, ours_comp, ours_random_all_suff, ours_random_all_comp = get_one_line_for_one_FA(model_name, "ours")

    suff_df = pd.DataFrame([signed_suff, integrated_suff, shap_suff, rollout_suff, last_suff, all_suff, ours_suff], columns=['FAs', 'Soft Suff'])
    comp_df = pd.DataFrame([signed_comp, integrated_comp, shap_comp, rollout_comp, last_comp, all_comp, ours_comp], columns=['FAs', 'Soft Comp'])

    # random_suff_df = pd.DataFrame([random_signed_suff, random_integrated_suff, random_shap_suff, random_rollout_suff, random_last_suff, random_all_suff, ours_random_all_suff], columns=['FAs', 'Soft Suff'])
    # random_comp_df = pd.DataFrame([random_signed_comp, random_integrated_comp, random_shap_comp, random_rollout_comp, random_last_comp, random_all_comp, ours_random_all_comp], columns=['FAs', 'Soft Comp'])


    # print()
    # print(f' =======>   {model_name} =======')
    # final_suff_df = minus_and_save(suff_df, random_suff_df, 'suff')
    # final_comp_df = minus_and_save(comp_df, random_comp_df, 'comp')


    stacked_df = pd.concat([suff_df, comp_df])
    stacked_df.to_csv(f'evaluation_results/summary/{model_name}/{hyper}_3.csv')

    print(' Done')
    return stacked_df, ours_random_all_suff[-1], ours_random_all_comp[-1]


model_name_dict = { 'gpt2':'GPT2 354M', 'gpt2_xl': 'GPT2 1.5B', 'gpt6b': 'GPT-J 6B', \
                    'OPT350M': 'OPT 350M', 'OPT1B':'OPT 1.3B', 'OPT6B':'OPT 6.7B', \
                    'GradxEmb':'Grad x Emb', }


all_results = []

randomFA_suff_of_models = []
randomFA_comp_of_models = []
for model_name in ['OPT350M', 'gpt2', 'gpt2_xl', 'OPT1B', 'OPT6B', 'gpt6b']: #'gpt6b', 
    temp, ours_random_all_suff, ours_random_all_comp = get_one_model(model_name)
    randomFA_suff_of_models.append(ours_random_all_suff)
    randomFA_comp_of_models.append(ours_random_all_comp)
    temp['Model'] = model_name
    temp.replace(model_name_dict,inplace=True)
    all_results.append(temp)


df = pd.concat(all_results)
df.to_csv('evaluation_results/summary/token_level_all.csv')


suff = df[["FAs","Soft Suff","Model"]]
comp = df[["FAs","Soft Comp","Model"]]


plt.figure(figsize=(22, 22))
fig, axs = plt.subplots(nrows=3, ncols=1, sharex=False, ) #squeeze=True,
#fig.title('Wikitext sentence-level faithfulness')

plt.subplot(3,1,3)  # row colum
axs[2].set_visible(False)

plt.subplot(3,1,1)  # row colum
sns.barplot(x="Model", y="Soft Comp", hue="FAs", data=comp, errorbar=None, width= 0.6,
            order=['OPT 350M','OPT 1.3B', 'OPT 6.7B','GPT2 354M', 'GPT2 1.5B', 'GPT-J 6B'], ax=axs[0]) #
plt.xlabel('Models', fontweight='bold')
for i in range(6):
    print(randomFA_suff_of_models[i])
    axs[0].hlines(y = randomFA_suff_of_models[i], xmin = i-0.5, xmax = i+0.5,
    color = 'black')

plt.subplot(3,1,2)  # row colum
sns.barplot(x="Model", y="Soft Suff", hue="FAs", data=suff, errorbar=None,  width= 0.6,
            order=['OPT 350M','OPT 1.3B', 'OPT 6.7B','GPT2 354M', 'GPT2 1.5B', 'GPT-J 6B']) # , height=8
plt.xlabel('Models', fontweight='bold')
for i in range(6):
    axs[0].hlines(y = randomFA_comp_of_models[i], xmin = i-0.5, xmax = i+0.5,
    color = 'black')



handles, labels = axs[0].get_legend_handles_labels()

fig.legend(handles[:8], labels[:8], ncol=4, loc='center', bbox_to_anchor=(0.5, 0.21), fontsize=9) # 00 0.4 middle 0.8 top
# fig.legend(nrow=1, loc='lower right', bbox_to_anchor=(1.19, 0.1)) #
plt.legend()
axs[0].get_legend().remove()
axs[1].get_legend().remove()
# Add xticks on the middle of the group bars

fig.suptitle('LongRA (token-level)', fontsize=15)
fig.tight_layout() 
plt.show()

plt.savefig(f"./evaluation_results/summary/token_{hyper}.png",bbox_inches='tight')
print(f"./evaluation_results/summary/token_{hyper}.png")

