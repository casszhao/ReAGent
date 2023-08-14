from transformers import AutoModelForCausalLM, AutoTokenizer
from rationalizer.aggregate_rationalizer import AggregateRationalizer
from rationalizer.importance_score_evaluator.delta_prob import DeltaProbImportanceScoreEvaluator
from rationalizer.stopping_condition_evaluator.top_k import TopKStoppingConditionEvaluator
from rationalizer.token_replacement.token_replacer.uniform import UniformTokenReplacer
from rationalizer.token_replacement.token_sampler.postag import POSTagTokenSampler

import argparse
import torch
import seaborn
import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy

# construct rationalizer


parser = argparse.ArgumentParser()

parser.add_argument("--model_name", 
                    type=str,
                    default='gpt2-medium', #'KoboldAI/OPT-6.7B-Erebus', #'gpt2-medium', 
                    help="path for storing the importance scores extracted") # TODO

args = parser.parse_args()


device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir='../../cache')
model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir='../../cache').to(device)


# input_string = "Give me my bottle back. Please paraphrase this sentence::"
# max_length = (input_string.split(' ')) * 2
# # generate prediction 
# input_ids = tokenizer(input_string, return_tensors='pt')['input_ids'][0].to(model.device)
# generated_ids = model.generate(input_ids=torch.unsqueeze(input_ids, 0), do_sample=False, pad_token_id = 50256 )[0]
# generated_texts = [ tokenizer.decode(token) for token in generated_ids ]
# print(f"==>> {generated_texts}")
# quit()

rational_size = 5
rationale_size_ratio = None
max_steps = 3000 
replace_ratio_for_update = 0.3 
topk_for_stopping=5 # when you want to shorter rationale (greater importance for less tokens), give it a small number
batch=5

token_sampler = POSTagTokenSampler(tokenizer=tokenizer, device=device)

stopping_condition_evaluator = TopKStoppingConditionEvaluator(model=model, 
                                                                token_sampler=token_sampler, 
                                                                top_k=topk_for_stopping, 
                                                                top_n=rational_size, 
                                                                top_n_ratio=rationale_size_ratio, 
                                                                tokenizer=tokenizer
                                                            )

importance_score_evaluator = DeltaProbImportanceScoreEvaluator(model=model, 
                                                                tokenizer=tokenizer, 
                                                                token_replacer=UniformTokenReplacer(
                                                                token_sampler=token_sampler, 
                                                                ratio=replace_ratio_for_update
                                                                ),
                                                                stopping_condition_evaluator=stopping_condition_evaluator,
                                                                max_steps=max_steps
                                                            )

rationalizer = AggregateRationalizer(importance_score_evaluator=importance_score_evaluator,
                                        batch_size=batch,
                                        overlap_threshold=2,
                                        overlap_strict_pos=True,
                                        top_n=rational_size, 
                                        top_n_ratio=rationale_size_ratio
                                    )




#input_string = "Model explanation is a difficult# "
#input_string ="Drew appeared busy yet comfortable as she was spotted on Wednesday, on the set of a"
input_string ="When my flight landed in, I converted my currency and slowly fell asleep. (I had a terrifying dream about my grandmother, but that's a story for another time). I was,"

max_length = 55

# generate prediction 
input_ids = tokenizer(input_string, return_tensors='pt')['input_ids'][0].to(model.device)

generated_ids = model.generate(input_ids=torch.unsqueeze(input_ids, 0), max_length=max_length, do_sample=False)[0]

generated_texts = [ tokenizer.decode(token) for token in generated_ids ]
print(f'generated full sequence --> {generated_texts}')



print(f"len of input_ids ==>> {input_ids}")
print(f"generated_ids ==>> {generated_ids}")

immediate_probability = model(input_ids=torch.unsqueeze(input_ids, 0))[0][:, -1, :]
immediate_probability_after_softmax = torch.softmax(immediate_probability, -1) 
next_token_index =  len(input_ids) - len(generated_ids)
next_token_id = generated_ids[next_token_index]

print(f"==>> {next_token_id}, {tokenizer.decode(next_token_id)}")


print(f"immediate_probability ==>> {immediate_probability}")
print(f"immediate_probability_after_softmax ==>> {immediate_probability_after_softmax}")
print("".center(50, "-"))
# print(immediate_probability_after_softmax.size())
# print(len(tokenizer))
# print("".center(50, "-"))

token_prob = immediate_probability_after_softmax[0,next_token_id]
print(f"token_prob ==>> {token_prob}")
print(immediate_probability_after_softmax[0,next_token_id])
print(immediate_probability_after_softmax[0,21891])
quit()



# rationalize each generated token

importance_scores = []
importance_score_map = torch.zeros([generated_ids.shape[0] - input_ids.shape[0], generated_ids.shape[0] - 1])

for target_pos in torch.arange(input_ids.shape[0], generated_ids.shape[0]):
    # extract target
    target_id = generated_ids[target_pos]

    # rationalization
    pos_rational = rationalizer.rationalize(torch.unsqueeze(generated_ids[:target_pos], 0), torch.unsqueeze(target_id, 0))[0]

    ids_rational = generated_ids[pos_rational]
    text_rational = [ tokenizer.decode([id_rational]) for id_rational in ids_rational ]

    importance_score_map[target_pos - input_ids.shape[0], :target_pos] = rationalizer.mean_important_score

    print(f'{target_pos} / {generated_ids.shape[0]}')
    print(f'Target word     --> {tokenizer.decode(target_id)}', )
    print(f"Rational pos    --> {pos_rational}")
    print(f"Rational text   --> {text_rational}")

    print()

mask_bloo = importance_score_map == 0

importance_score_map =numpy.array([numpy.array(xi) for xi in importance_score_map])
mask_bloo =numpy.array([numpy.array(xi) for xi in mask_bloo])


seaborn.set(rc={ 'figure.figsize': (14, 7) })
s = seaborn.heatmap(
    importance_score_map, 
    xticklabels=generated_texts[:-1], 
    yticklabels=generated_texts[input_ids.shape[0]:], 
    annot=True, 
    mask=mask_bloo, 
    square=True,
    linewidth=.5,
    cbar_kws={'label': 'Importance'}
    )
plt.title(str(args.model_name))
s.set_ylabel('Predicted Word')
s.set_xlabel('Prompt (Input)')

#plt.tight_layout()
scatter_fig = s.get_figure()


try: saved_name = str(args.model_name).replace("/", "_")
except: saved_name=str(args.model_name)
scatter_fig.savefig(f'./visual/{saved_name}.png')
print(' done')
