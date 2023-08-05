import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = 'wikitext' # tellmewhy wikitext

model_name_dict = { 'gpt2':'GPT2 354M', 'gpt2_xl': 'GPT2 1.5B', 'gpt6b': 'GPT-J 6B', \
                    'OPT350M': 'OPT 350M', 'OPT1B':'OPT 1.3B', 'OPT6B':'OPT 6.7B', \
                    'GradxEmb':'Grad x Emb', }

df = pd.read_csv('./evaluation_results/summary/benchmark/ALL.csv')
# "FAs","Soft Suff","Soft Comp","Model","Data"
df.replace(model_name_dict,inplace=True)

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
sns.barplot(x="Model", y="Soft Comp", hue="FAs", data=comp, errorbar=None, width= 0.6,
            order=['OPT 350M','OPT 1.3B', 'OPT 6.7B','GPT2 354M', 'GPT2 1.5B'])
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

fig.suptitle(f' {dataset.capitalize()}', fontsize=15)
fig.tight_layout() 
plt.show()

plt.savefig(f"./evaluation_results/summary/benchmark/{dataset}_sentence.png",bbox_inches='tight')

