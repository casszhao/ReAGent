
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn

import numpy as np

import torch
import torch.nn as nn


# from captum.attr import IntegratedGradients
# from transformers import AutoModel, AutoTokenizer

# model_name = 'andi611/distilbert-base-uncased-ner-agnews'
# tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='cache')
# model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir='cache').to('cuda')


# gradients=[     0.05368266, 0.07159787, 0.04512938, 0.02630155,
#                 0.1079347 , 0.06782196, 0.25342748, 0.02630155, 0.00085337,
#                 0.06320742, 0.04962058, 0.00693226, 0.01740681, 0.08596288,
#                 0.03031706, 0.01062884, 0.04063026, 0.00715546, 0.0006621 ,
#                 0.03124025, 0.03197142, 0.0327077 , 0.02400859, 0.0037614 ,
#                 0.08822088, 0.00609987, 0.02714362, 0.02335986, 0.03031706,
#                 0.01209414, 0.03124025, 0.01928692, 0.03031706, 0.00906844,
#                 0.01563995, 0.06653544, 0.00232135, 0.09721994]

# # points_text=["with", "a", "story", "bizarre",  "and", "mysterious" , "as" , "this" , "you", "do" ,"n" ,"t" ,"want" ,"to" ,"be" ,"worrying", "about" , "..." , "...", "...", "...", "...", "that", "door"]
# # points_index=[ 0,     2,    3,        6,       7,          9,        10,      11,     12,    14,  16,   18,    20,    21,  22,    23,        25,      26,     27,     28,    29,    31,  gradients_len-2, gradients_len]
# points_text=[' "with a story as bizarre and mysterious as this you do n t ..... have the courage to knock on that door " ']
# points_index=[ 19]



text = ["Model ", "  explanation", "  is", " a", "  hard", "  one", "  to", "  explain", "  ."]
gradients = [0.301,     0.221,   0.011,  0.021,  0.207,    0.093,    0.044,   0.001,   0.101]

plt.rcParams["figure.figsize"] = 12,2

fig, ax = plt.subplots()
seaborn.heatmap([gradients], ax=ax, cbar_kws={'label': 'Importance', 'aspect':5},
                annot=True, annot_kws={'size': 15},
               # xticklabels=text,
                )


ax.set(yticklabels=[])
ax.set_xticklabels(text, fontsize=14)
# ax.set(xticks=range(len(text)))
# ax.set(xticklabels=text) 

#ax.set_xticklabels(text, fontsize=12) 

plt.xlabel('Input', fontweight='bold', fontsize=12)
plt.ylabel('Prediction\n\nLabel:Sci/Tech', fontweight='bold', fontsize=12)
plt.gca().yaxis.label.set(rotation='horizontal', ha='right')

# plt.setp(ax.get_xticklabels(), visible=False)
# plt.setp(ax.get_xticks(), visible=False)
# ax.tick_params(axis='y', which='both') #

plt.title('BERT Fine-tuned on AGnews for Topic Classification')
# ax.legend(title="Importance\n scores", loc='upper left', bbox_to_anchor=(1, -0.01),
#           fancybox=False) # (1.05 1) borderaxespad=0, 

plt.tight_layout()
plt.savefig("./one_line_heatmap.png",bbox_inches='tight')
print("./one_line_heatmap.png")

quit()
from transformers import AutoModelForCausalLM, AutoModelWithLMHead, AutoTokenizer
import os


input_string = "Mary always bragged about how good she was at bowling. Mary's co-worker Steve challenged her to a bowling competition. Steve beat Mary four times in a row. Mary sat on the steps with a sad look on her face. How was she to know that Steve was a college bowling champion? Why did Steve beat Mary?"


model = AutoModelWithLMHead.from_pretrained("gpt2-medium").to('cuda')
tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")

input_ids = tokenizer(input_string, return_tensors='pt')['input_ids'].to(model.device)
generated_input = model.generate(input_ids=input_ids, do_sample=False, max_new_tokens=10) 
# print(f"{generated_input}")
# generated_logits = model(input_ids=input_ids)['logits']


# print(' ---- small ')
# print(generated_logits)

print(tokenizer.decode(generated_input[0]))