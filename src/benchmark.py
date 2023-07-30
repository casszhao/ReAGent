
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
from rationalization.rationalizer.aggregate_rationalizer import AggregateRationalizer
from rationalization.rationalizer.importance_score_evaluator.delta_prob import DeltaProbImportanceScoreEvaluator
from rationalization.rationalizer.stopping_condition_evaluator.top_k import TopKStoppingConditionEvaluator
from rationalization.rationalizer.token_replacement.token_replacer.uniform import UniformTokenReplacer
from rationalization.rationalizer.token_replacement.token_sampler.postag import POSTagTokenSampler

from evaluation.evaluator.soft_norm_sufficiency import SoftNormalizedSufficiencyEvaluator
from evaluation.evaluator.soft_norm_comprehensiveness import SoftNormalizedComprehensivenessEvaluator
import seaborn

import csv

# config global
device = "cuda"
output_dir = "benchmark_results/test"
input_file = "data/benchmark/wikitext.txt"

# config generation
gen_length = 10

# config evaluator
metric_stride = 2

# init input
with open(input_file, "r") as in_f:
    input_text_list = in_f.read().splitlines()

# init model

tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
model = AutoModelForCausalLM.from_pretrained("gpt2-medium").to(device)

# init evaluator

soft_norm_suff_evaluator = SoftNormalizedSufficiencyEvaluator(model)
soft_norm_comp_evaluator = SoftNormalizedComprehensivenessEvaluator(model)

# init rationalizer

rational_size = 5
rational_size_ratio = None

token_sampler = POSTagTokenSampler(tokenizer=tokenizer, device=device)

stopping_condition_evaluator = TopKStoppingConditionEvaluator(
    model=model, 
    token_sampler=token_sampler, 
    top_k=10, 
    top_n=rational_size, 
    top_n_ratio=rational_size_ratio, 
    tokenizer=tokenizer
)

importance_score_evaluator = DeltaProbImportanceScoreEvaluator(
    model=model, 
    tokenizer=tokenizer, 
    token_replacer=UniformTokenReplacer(
        token_sampler=token_sampler, 
        ratio=0.3
    ),
    stopping_condition_evaluator=stopping_condition_evaluator,
    max_steps=3000
)

rationalizer = AggregateRationalizer(
    importance_score_evaluator=importance_score_evaluator,
    batch_size=5,
    overlap_threshold=2,
    overlap_strict_pos=True,
    top_n=rational_size, 
    top_n_ratio=rational_size_ratio
)



for i, input_text in enumerate(input_text_list):

    # generation

    input_ids = tokenizer(input_text, return_tensors='pt')['input_ids'][0].to(model.device)
    max_length = input_ids.shape[0] + gen_length
    generated_ids = model.generate(input_ids=torch.unsqueeze(input_ids, 0), max_length=max_length, do_sample=False)[0]
    generated_texts = [ tokenizer.decode(token) for token in generated_ids ]
    print(f'generated full sequence --> {generated_texts}')

    # rationalization

    importance_scores = []
    importance_score_map = torch.zeros([generated_ids.shape[0] - input_ids.shape[0], generated_ids.shape[0] - 1], device=device)

    for target_pos in torch.arange(input_ids.shape[0], generated_ids.shape[0]):
        
        # extract target
        target_id = generated_ids[target_pos]

        # rationalization
        pos_rational = rationalizer.rationalize(torch.unsqueeze(generated_ids[:target_pos], 0), torch.unsqueeze(target_id, 0))[0]

        ids_rational = generated_ids[pos_rational]
        text_rational = [ tokenizer.decode([id_rational]) for id_rational in ids_rational ]

        importance_score_map[target_pos - input_ids.shape[0], :target_pos] = rationalizer.mean_important_score

        print(f'{target_pos + 1} / {generated_ids.shape[0]}')
        print(f'Target word     --> {tokenizer.decode(target_id)}', )
        print(f"Rational pos    --> {pos_rational}")
        print(f"Rational text   --> {text_rational}")
        print()

    # evaluation
    
    norm_suff_all = []
    norm_comp_all = []
    target_token_all = []

    table_details = [ ["target_pos", "target_token", "norm_suff", "norm_comp"] ]

    for target_pos in torch.arange(input_ids.shape[0], generated_ids.shape[0], metric_stride):

        target_token = tokenizer.decode(generated_ids[target_pos])
        target_token_all.append(target_token)

        input_ids_step = torch.unsqueeze(generated_ids[:target_pos], 0)
        target_id_step = torch.unsqueeze(generated_ids[target_pos], 0)
        importance_score_step = torch.unsqueeze(importance_score_map[target_pos - input_ids.shape[0], :target_pos], 0)

        norm_suff = soft_norm_suff_evaluator.evaluate(input_ids_step, target_id_step, importance_score_step)
        norm_suff_all.append(norm_suff)

        norm_comp = soft_norm_comp_evaluator.evaluate(input_ids_step, target_id_step, importance_score_step)
        norm_comp_all.append(norm_comp)

        table_details.append([target_pos.item() + 1, target_token, norm_suff.item(), norm_comp.item()])
        print(f"target_pos: {target_pos + 1}, target_token: {target_token}, norm_suff: {norm_suff}, norm_comp: {norm_comp}")

    norm_suff_mean = torch.mean(torch.tensor(norm_suff_all, device=device))
    norm_comp_mean = torch.mean(torch.tensor(norm_comp_all, device=device))

    print(f"norm_suff_mean: {norm_suff_mean}, norm_comp_mean: {norm_comp_mean}")

    # export generated_texts

    with open(os.path.join(output_dir, f'{i}_output.txt'), 'w') as outfile:
        outfile.writelines(generated_texts)

    # export table

    table_mean = [
        [ "norm_suff_mean", "norm_comp_mean", "target_tokens" ],
        [ norm_suff_mean.item(), norm_comp_mean.item(), "$".join(target_token_all) ]
    ]

    with open(os.path.join(output_dir, f'{i}_details.csv'), 'w', newline='') as csvfile:
        csvWriter = csv.writer(csvfile)
        csvWriter.writerows(table_details)
    
    with open(os.path.join(output_dir, f'{i}_mean.csv'), 'w', newline='') as csvfile:
        csvWriter = csv.writer(csvfile)
        csvWriter.writerows(table_mean)

    # export plot
    seaborn.set(rc={ 'figure.figsize': (30, 10) })
    s = seaborn.heatmap(
        importance_score_map.cpu(), 
        xticklabels=generated_texts[:-1], 
        yticklabels=generated_texts[input_ids.shape[0]:], 
        annot=True, 
        square=True)
    s.set_xlabel('Importance distribution')
    s.set_ylabel('Target')
    fig = s.get_figure()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{i}_dist.png'))
    fig.clf()
