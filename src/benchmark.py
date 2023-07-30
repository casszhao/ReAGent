
import argparse
import json
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

parser = argparse.ArgumentParser()

parser.add_argument("--rationalization-config", 
                    type=str,
                    default="config/test.json",
                    help="") # TODO

args = parser.parse_args()

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

with open(args.rationalization_config) as f_config:
    rationalization_config = json.load(f_config)

importance_score_evaluator_type = rationalization_config["importance_score_evaluator"]["type"]

if importance_score_evaluator_type == "replacing":

    replacing_type = rationalization_config["importance_score_evaluator"]["replacing"]["replacing"]["type"]
    if replacing_type == "uniform":
        from rationalization.rationalizer.token_replacement.token_sampler.uniform import \
            UniformTokenSampler
        token_sampler = UniformTokenSampler(tokenizer)
    elif replacing_type == "inferential":
        from rationalization.rationalizer.token_replacement.token_sampler.inferential import \
            InferentialTokenSampler
        token_sampler = InferentialTokenSampler(tokenizer=tokenizer, model=model)
    elif replacing_type == "postag":
        from rationalization.rationalizer.token_replacement.token_sampler.postag import \
            POSTagTokenSampler
        token_sampler = POSTagTokenSampler(tokenizer=tokenizer, device=device)
    else:
        raise ValueError(f"Invalid replacement_sampling: {replacing_type}")
    
    stopping_condition_type = rationalization_config["importance_score_evaluator"]["replacing"]["stopping_condition"]["type"]
    if stopping_condition_type == "top_k":
        from rationalization.rationalizer.stopping_condition_evaluator.top_k import \
            TopKStoppingConditionEvaluator
        top_k=rationalization_config["importance_score_evaluator"]["replacing"]["stopping_condition"]["top_k"]["tolerance"]
        top_n=rationalization_config["rational"]["size"]
        top_n_ratio=rationalization_config["rational"]["size_ratio"]
        stopping_condition_evaluator = TopKStoppingConditionEvaluator(
            model=model, 
            token_sampler=token_sampler, 
            top_k=top_k, 
            top_n=top_n, 
            top_n_ratio=top_n_ratio, 
            tokenizer=tokenizer
        )
        #output_dir = output_dir + f'/top{top_k}_' # by cass

    elif stopping_condition_type == "dummy":
        from rationalization.rationalizer.stopping_condition_evaluator.dummy import \
            DummyStoppingConditionEvaluator
        stopping_condition_evaluator = DummyStoppingConditionEvaluator()
    else:
        raise ValueError(f"Invalid stopping_condition: {stopping_condition_type}")

    evaluator_type = rationalization_config["importance_score_evaluator"]["replacing"]["optimization"]["type"]
    if evaluator_type == 'delta_probability':
        from rationalization.rationalizer.importance_score_evaluator.delta_prob import \
            DeltaProbImportanceScoreEvaluator
        from rationalization.rationalizer.token_replacement.token_replacer.uniform import \
            UniformTokenReplacer
        replacing_ratio=rationalization_config["importance_score_evaluator"]["replacing"]["optimization"]["delta_probability"]["replacing_ratio"]
        max_steps=rationalization_config["importance_score_evaluator"]["replacing"]["optimization"]["delta_probability"]["max_steps"]
        output_dir = output_dir + f'replace{replacing_ratio}_max{max_steps}' # by cass
        importance_score_evaluator = DeltaProbImportanceScoreEvaluator(
            model=model, 
            tokenizer=tokenizer, 
            token_replacer=UniformTokenReplacer(
                token_sampler=token_sampler, 
                ratio=replacing_ratio
            ),
            stopping_condition_evaluator=stopping_condition_evaluator,
            max_steps=max_steps
        )
    elif evaluator_type == 'bayesian_optimization':
        from rationalization.rationalizer.importance_score_evaluator.bayesian_opti import \
            BayesianOptimizationImportanceScoreEvaluator
        from rationalization.rationalizer.token_replacement.token_replacer.ranking import \
            RankingTokenReplacer
        importance_score_evaluator = BayesianOptimizationImportanceScoreEvaluator(
            model=model, 
            tokenizer=tokenizer, 
            token_replacer=RankingTokenReplacer(
                token_sampler=token_sampler, 
                top_n=rationalization_config["rational"]["size"], 
                top_n_ratio=rationalization_config["rational"]["size_ratio"], 
            ),
            stopping_condition_evaluator=stopping_condition_evaluator,
            sample_multiplier=rationalization_config["importance_score_evaluator"]["replacing"]["optimization"]["bayesian_optimization"]["sampling"]["multiplier"],
            sample_increment=rationalization_config["importance_score_evaluator"]["replacing"]["optimization"]["bayesian_optimization"]["sampling"]["increment"],
            training_config=rationalization_config["importance_score_evaluator"]["replacing"]["optimization"]["bayesian_optimization"]["training"],
            optimizing_config=rationalization_config["importance_score_evaluator"]["replacing"]["optimization"]["bayesian_optimization"]["optimizing"]
        )
    else:
        raise ValueError(f"Invalid evaluator-type: {evaluator_type}")

elif importance_score_evaluator_type == "attention":
    from rationalization.rationalizer.importance_score_evaluator.attention import \
        AttentionImportanceScoreEvaluator
    importance_score_evaluator = AttentionImportanceScoreEvaluator(
        model=model,
        tokenizer=tokenizer,
        attn_type=rationalization_config["importance_score_evaluator"]["attention"]["type"]
    )
elif importance_score_evaluator_type == "gradient":
    from rationalization.rationalizer.importance_score_evaluator.grad import \
        GradientImportanceScoreEvaluator
    importance_score_evaluator = GradientImportanceScoreEvaluator(
        model=model,
        tokenizer=tokenizer,
        grad_type=rationalization_config["importance_score_evaluator"]["gradient"]["type"]
    )
elif importance_score_evaluator_type == "inseq":
    from rationalization.rationalizer.importance_score_evaluator.inseq import \
        InseqImportanceScoreEvaluator
    importance_score_evaluator = InseqImportanceScoreEvaluator(
        model=model,
        tokenizer=tokenizer,
        method=rationalization_config["importance_score_evaluator"]["inseq"]["type"],
        attribute_params=rationalization_config["importance_score_evaluator"]["inseq"]["attribute_params"]
    )
else:
    raise ValueError(f"Invalid importance_score_evaluator_type {importance_score_evaluator_type}")
    
rationalizer_type = rationalization_config["rationalizer"]["type"]
if rationalizer_type == "sampling":
    from rationalization.rationalizer.sample_rationalizer import SampleRationalizer
    rationalizer = SampleRationalizer(
        importance_score_evaluator=importance_score_evaluator, 
        top_n=rationalization_config["rational"]["size"], 
        top_n_ratio=rationalization_config["rational"]["size_ratio"]
    )
elif rationalizer_type == "aggregation":
    from rationalization.rationalizer.aggregate_rationalizer import AggregateRationalizer
    rationalizer = AggregateRationalizer(
        importance_score_evaluator=importance_score_evaluator,
        batch_size=rationalization_config["rationalizer"]["aggregation"]["batch_size"],
        overlap_threshold=rationalization_config["rationalizer"]["aggregation"]["overlap_threshold"],
        overlap_strict_pos=rationalization_config["rationalizer"]["aggregation"]["overlap_strict_pos"],
        top_n=rationalization_config["rational"]["size"], 
        top_n_ratio=rationalization_config["rational"]["size_ratio"]
    )
else:
    raise ValueError(f"Invalid rationalizer_type {rationalizer_type}")



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
