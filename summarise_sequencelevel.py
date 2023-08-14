import pandas as pd
import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math


model_name_dict = { 'gpt2':'GPT2 354M', 'gpt2_xl': 'GPT2 1.5B', 'gpt6b': 'GPT-J 6B', \
                    'OPT350M': 'OPT 350M', 'OPT1B':'OPT 1.3B', 'OPT6B':'OPT 6.7B', \
                    'GradxEmb':'Grad x Emb', }

def get_one_line_for_one_FA(model_name, FA_name, task_name):
    print(' ====> ', FA_name)
    eva_output_dir=f"evaluation_results/benchmark/{model_name}_{FA_name}/{task_name}/"
    directory = os.fsencode(eva_output_dir)
    suff_mean = 0
    comp_mean = 0
    random_suff_mean = 0
    random_comp_mean = 0

    if FA_name == 'norm': lis =['Grad norms']
    elif FA_name == 'input_x_gradient': lis =['GradxEmb']
    elif FA_name == 'integrated_gradients': lis =['Integrated Grad']
    else: lis = [FA_name.replace('_', ' ').title()]
    
    len = 0
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        
        if filename.endswith("mean.csv"): 
            len +=1
            faithful_results = pd.read_csv(eva_output_dir+filename) # one data
            suff_mean += faithful_results['norm_suff_mean'][0]
            random_suff_mean += faithful_results['random_suff_mean'][0]
            comp_mean += faithful_results['norm_comp_mean'][0]
            random_comp_mean += faithful_results['random_comp_mean'][0]
            continue
        else:continue

    # print(suff_mean-random_suff_mean, comp_mean-random_comp_mean)
    # lis.append((suff_mean-random_suff_mean)/len)
    # lis.append((comp_mean-random_comp_mean)/len)

    suff = math.log(suff_mean/random_suff_mean)
    comp = math.log(comp_mean/random_comp_mean)

    lis.append(suff)
    lis.append(comp)
    return lis



all_results = []
for model_name in ["gpt6b", "OPT350M", "gpt2", "gpt2_xl", "OPT1B", "OPT6B"]: # "gpt2","gpt2_xl", "OPT1B", "OPT6B"
# "gpt6b", "OPT350M", 
    for dataset in ['tellmewhy', 'wikitext']: # 
        print()
        print()
        print(f' ============== {model_name},  {dataset}  ============== ')
        # try: norm = get_one_line_for_one_FA(model_name, "norm", dataset)
        # except: norm = None
        signed = get_one_line_for_one_FA(model_name, "input_x_gradient", dataset)
        integrated = get_one_line_for_one_FA(model_name, "integrated_gradients", dataset)
        gradient_shap = get_one_line_for_one_FA(model_name, "gradient_shap", dataset)

        rollout_attention = get_one_line_for_one_FA(model_name, "attention_rollout", dataset)
        last_attention = get_one_line_for_one_FA(model_name, "attention_last", dataset)
        attention = get_one_line_for_one_FA(model_name, "attention", dataset)
        ours = get_one_line_for_one_FA(model_name, "ours", dataset)

        df = pd.DataFrame([signed, integrated, gradient_shap,\
                            rollout_attention, last_attention, attention, ours], columns=['FAs', 'Soft Suff', 'Soft Comp'])
        print(df)
        
        os.makedirs(f'evaluation_results/summary/benchmark/{dataset}/', exist_ok=True)
        df.to_csv(f'evaluation_results/summary/benchmark/{dataset}/{model_name}_{dataset}.csv')
        df['Model'] = model_name
        df['Data'] = dataset
        all_results.append(df)


df = pd.concat(all_results)
df.to_csv(f'evaluation_results/summary/benchmark/ALL.csv')
df.replace(model_name_dict,inplace=True)


#dataset = 'wikitext' # tellmewhy wikitext
for dataset in ['wikitext', 'tellmewhy']: # wikitext

    # "FAs","Soft Suff","Soft Comp","Model","Data"
    

    select_data = df.loc[df['Data'] == dataset]
    suff = select_data[["FAs","Soft Suff","Model"]]
    comp = select_data[["FAs","Soft Comp","Model"]]



    #sns.set(style="darkgrid")

    plt.figure(figsize=(22, 22))
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=False, ) #squeeze=True,
    #fig.title('Wikitext sentence-level faithfulness')

    plt.subplot(3,1,3)  # row colum
    axs[2].set_visible(False)

    plt.subplot(3,1,1)  # row colum
    sns.barplot(x="Model", y="Soft Comp", hue="FAs", data=comp, #errorbar=None, #width= 0.6,
                order=['OPT 350M','OPT 1.3B', 'OPT 6.7B','GPT2 354M', 'GPT2 1.5B', 'GPT-J 6B'])
    plt.xlabel('Models', fontweight='bold')

    plt.subplot(3,1,2)  # row colum
    sns.barplot(x="Model", y="Soft Suff", hue="FAs", data=suff, #errorbar=None,  width= 0.6,
                order=['OPT 350M','OPT 1.3B', 'OPT 6.7B','GPT2 354M', 'GPT2 1.5B', 'GPT-J 6B']) # , height=8
    plt.xlabel('Models', fontweight='bold')



    handles, labels = axs[0].get_legend_handles_labels()

    fig.legend(handles[:8], labels[:8], ncol=4, loc='center', bbox_to_anchor=(0.5, 0.21), fontsize=9) # 00 0.4 middle 0.8 top
    # fig.legend(nrow=1, loc='lower right', bbox_to_anchor=(1.19, 0.1)) #
    plt.legend()
    axs[0].get_legend().remove()
    axs[1].get_legend().remove()
    # Add xticks on the middle of the group bars

    fig.suptitle(f' {dataset.capitalize()} (sequence-level)', fontsize=15)
    fig.tight_layout() 
    plt.show()

    plt.savefig(f"./evaluation_results/summary/benchmark/{dataset}_sentence.png",bbox_inches='tight')
    print(f"saving at ===>  ./evaluation_results/summary/benchmark/{dataset}_sentence.png")