import pandas as pd
import os 
import numpy as np
import math
import re






def get_step_faithful(model_short_name):
# Define the regular expression pattern
    pattern = r'evaluated in (\d+) steps'
    steps_list = []

    # Open and read the .log file

    if os.path.exists(f'logs/analogies/{model_short_name}_ours/top3_replace0.3_max3000_batch5run.log'): 
        hyper='top3_replace0.3_max3000_batch5'
        print(' defined hyper')
    elif os.path.exists(f'logs/analogies/{model_short_name}_ours/top3_replace0.3_max3000_batch10run.log'): 
        hyper='top3_replace0.3_max3000_batch10'
        print(' defined hyper *2 ')
    else:print(model_short_name)


    with open(f'logs/analogies/{model_short_name}_ours/{hyper}run.log', 'r') as file:
            for line in file:
            # Search for the pattern in the line
                match = re.search(pattern, line)
                if match:
                    # Extract the number from the match and convert it to an integer
                    number = int(match.group(1))
                    steps_list.append(number)

        

    mean = np.mean(steps_list)
    std_dev = np.std(steps_list)




    faithful_results = pd.read_csv(f'evaluation_results/analogies/{model_short_name}_ours/{hyper}/mean_1.0.csv')

    suff = faithful_results['suff'][0]
    rand_suff = faithful_results['random_suff'][0]
    comp = faithful_results['comp'][0]
    rand_comp = faithful_results['random_comp'][0]

    final_suff = math.log(suff / rand_suff)
    final_comp = math.log(comp / rand_comp)
    return (mean, std_dev, final_suff, final_comp)



mean, std_dev, final_suff, final_comp = [], [], [], []




model_list = ['gpt2', 'gpt2_xl', 'gpt6b', 'OPT350M', 'OPT1B', 'OPT6B'] #    

model_list_for_df, method_for_df = [],[]
for model_short_name in model_list:
 #for method in ["inferential-m"]: # , "uniform"
    res = get_step_faithful(model_short_name)
    mean.append(res[0])
    std_dev.append(res[1])
    final_suff.append(res[2])
    final_comp.append(res[3])

    model_list_for_df.append(model_short_name)


model_name_dict = { 'gpt2':'GPT2 354M', 'gpt2_xl': 'GPT2 1.5B', 'gpt6b': 'GPT-J 6B', \
                    'OPT350M': 'OPT 350M', 'OPT1B':'OPT 1.3B', 'OPT6B':'OPT 6.7B', \
                    'inferential-m':'RoBERTa', 'uniform':'random', \
                    }


df = pd.DataFrame({'Model':model_list_for_df, 'Steps': mean, 'Steps std': std_dev, 'Suff': final_suff, 'Comp': final_comp})
df.replace(model_name_dict,inplace=True)
df = df.round(3)


def format_to_integer(value):
    return int(value)


# Apply the formatting functions to specific columns
df['Steps'] = df['Steps'].apply(format_to_integer)
df['Steps std'] = df['Steps std'].apply(format_to_integer)



print(df)
df.to_csv('expl.csv')