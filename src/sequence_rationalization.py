
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import torch
import json
from rationalization.rationalizer.aggregate_rationalizer import AggregateRationalizer
from rationalization.rationalizer.importance_score_evaluator.delta_prob import DeltaProbImportanceScoreEvaluator
from rationalization.rationalizer.stopping_condition_evaluator.top_k import TopKStoppingConditionEvaluator
from rationalization.rationalizer.token_replacement.token_replacer.uniform import UniformTokenReplacer
from rationalization.rationalizer.token_replacement.token_sampler.postag import POSTagTokenSampler

from evaluation.evaluator.soft_norm_sufficiency import SoftNormalizedSufficiencyEvaluator
from evaluation.evaluator.soft_norm_comprehensiveness import SoftNormalizedComprehensivenessEvaluator
import seaborn

import csv



'''
Example for python:

import os
os.environ['TRANSFORMERS_CACHE'] = '/blabla/cache/'
Example for bash:

export TRANSFORMERS_CACHE=/blabla/cache/
'''


parser = argparse.ArgumentParser()

parser.add_argument("--model", 
                    type=str,
                    default="EleutherAI/gpt-j-6b",
                    help="select from ===> facebook/opt-350m facebook/opt-1.3b KoboldAI/OPT-6.7B-Erebus \
                        gpt2-medium gpt2-xl EleutherAI/gpt-j-6b") 
parser.add_argument("--model_shortname", 
                    type=str,
                    default="gpt6b", 
                    help="select from ===> OPT350M gpt2 gpt2_xl gpt6b OPT350M OPT1B OPT6B ") 

parser.add_argument("--testing_data_name", 
                    type=str,
                    default="tellmewhy2",
                    help="select between wikitext and tellmewhy") 

parser.add_argument("--method", 
                    type=str,
                    default="gradient_shap", 
                    help="ours, like \
                    attention attention_last attention_rollout \
                    gradient_shap  integrated_gradients  input_x_gradient norm ") # TODO
## OPT6B cannot run norm
## gpt2 wikitext run gradient_shap, only a few data 
## gpt2 wikitext run input_x_gradient, only a few data 
## gpt2_xl wikitext run attention / gradient_shap / input_x_gradient, only a few data ==> raise ValueError("Start and end attribution positions cannot be the same.")ValueError: Start and end attribution positions cannot be the same.

parser.add_argument("--stride", 
                    type=int,
                    default=1, 
                    help="") # TODO
parser.add_argument("--max_new_tokens", 
                    type=int,
                    default=10, 
                    help="") # TODO

parser.add_argument("--cache_dir", 
                    type=str,
                    default='cache/',
                    help="store models")
# parser.add_argument("--data-dir", 
#                     type=str,
#                     default="data/gpt",
#                     help="") # TODO
parser.add_argument("--if_image", 
                    type=bool,
                    default=False,
                    help="") # TODO


args = parser.parse_args()
# config global
device = "cuda"

import os
os.environ['TRANSFORMERS_CACHE'] = '/cache/'

testing_data_name = args.testing_data_name

output_dir = f"evaluation_results/benchmark/{args.model_shortname}_{args.method}/{testing_data_name}"
input_file = f"data/benchmark/{testing_data_name}.txt"

os.makedirs(output_dir, exist_ok=True)


# init input
with open(input_file, "r") as in_f:
    input_text_list = in_f.read().splitlines()

# init model

tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir='/mnt/parscratch/users/cass/seq_rationales/cache/') # , padding_side='left'
model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir='/mnt/parscratch/users/cass/seq_rationales/cache/').to(device)

# init evaluator

soft_norm_suff_evaluator = SoftNormalizedSufficiencyEvaluator(model)
soft_norm_comp_evaluator = SoftNormalizedComprehensivenessEvaluator(model)

# init rationalizer

rational_size = 3
rational_size_ratio = None

# tested with 3 0.1 5000 5

stopping_top_k = 3
replacing = 0.1
max_step = 3000
batch = 3

if args.method == 'ours':
    
    token_sampler = POSTagTokenSampler(tokenizer=tokenizer, device=device)

    stopping_condition_evaluator = TopKStoppingConditionEvaluator(
    model=model, 
    token_sampler=token_sampler, 
    top_k=stopping_top_k, 
    top_n=rational_size, 
    top_n_ratio=rational_size_ratio, 
    tokenizer=tokenizer
    )

    importance_score_evaluator = DeltaProbImportanceScoreEvaluator(
        model=model, 
        tokenizer=tokenizer, 
        token_replacer=UniformTokenReplacer(
            token_sampler=token_sampler, 
            ratio=replacing
        ),
        stopping_condition_evaluator=stopping_condition_evaluator,
        max_steps=max_step
    )


    rationalizer = AggregateRationalizer(
        importance_score_evaluator=importance_score_evaluator,
        batch_size=batch,
        overlap_threshold=2,
        overlap_strict_pos=True,
        top_n=rational_size, 
        top_n_ratio=rational_size_ratio
    )

elif args.method == 'attention_last' or args.method == 'attention_rollout':
    from rationalization.rationalizer.importance_score_evaluator.attention import AttentionImportanceScoreEvaluator
    importance_score_evaluator = AttentionImportanceScoreEvaluator(
            model=model,
            tokenizer=tokenizer,
            attn_type= args.method.replace("attention_", "")
        )
    from rationalization.rationalizer.sample_rationalizer import SampleRationalizer
    rationalizer = SampleRationalizer(
        importance_score_evaluator=importance_score_evaluator,
        top_n=3,
        #top_n_ratio=
    )
# elif args.method == 'integrated_gradients':  # l2
#     from rationalization.rationalizer.importance_score_evaluator.grad import GradientImportanceScoreEvaluator
#     importance_score_evaluator = GradientImportanceScoreEvaluator(
#             model=model,
#             tokenizer=tokenizer,
#             grad_type= args.method
#         )
#     from rationalization.rationalizer.sample_rationalizer import SampleRationalizer
#     rationalizer = SampleRationalizer(
#         importance_score_evaluator=importance_score_evaluator,
#         top_n=3,
#         #top_n_ratio=
#     )
else: 
    # assert args.method in ['integrated_gradients', 'input_x_gradient', 'attention', 'gradient_shap'] # input_x_gradient = signed in self written
    from rationalization.rationalizer.importance_score_evaluator.inseq import InseqImportanceScoreEvaluator
    importance_score_evaluator = InseqImportanceScoreEvaluator(
        model=model, 
        tokenizer=tokenizer, 
        method=args.method, # integrated_gradients input_x_gradient attention
        attribute_params={
        #show_progress = False,
        }
    )


    from rationalization.rationalizer.sample_rationalizer import SampleRationalizer
    rationalizer = SampleRationalizer(
        importance_score_evaluator=importance_score_evaluator,
        top_n=3,
        #top_n_ratio=
    )



for i, input_text in enumerate(input_text_list):

    raw_dict = {}

    try:
        # generation

        input_ids = tokenizer(input_text, return_tensors='pt')['input_ids'][0].to(model.device)
        max_length = input_ids.shape[0] + args.max_new_tokens
        generated_ids = model.generate(input_ids=torch.unsqueeze(input_ids, 0), max_length=max_length, do_sample=False)[0]
        generated_texts = [ tokenizer.decode(token) for token in generated_ids ]
        print(f'generated full sequence --> {generated_texts}')

        # rationalization
        normalise_random = torch.nn.Softmax(dim=1) # For normalise random baseline

        importance_scores = []
        importance_score_map = torch.zeros([generated_ids.shape[0] - input_ids.shape[0], generated_ids.shape[0] - 1], device=device)

        skipped_pos = set()

        for target_pos in torch.arange(input_ids.shape[0], generated_ids.shape[0]):

            # extract target
            target_id = generated_ids[target_pos]

            try:

                # rationalization
                pos_rational = rationalizer.rationalize(torch.unsqueeze(generated_ids[:target_pos], 0), torch.unsqueeze(target_id, 0))[0]

            except ValueError as e:
                print(f'[Warn] failed on {i} - {target_pos}')
                import traceback
                traceback.print_exception(e)

                skipped_pos.add(target_pos)

                continue
            
            ids_rational = generated_ids[pos_rational]
            text_rational = [ tokenizer.decode([id_rational]) for id_rational in ids_rational ]

            if rationalizer.mean_important_score.shape[0] != target_pos:
                print(f'[Warn] failed on {i} - {target_pos}: length of importance score does not match')
                skipped_pos.add(target_pos)
                continue

            importance_score_map[target_pos - input_ids.shape[0], :target_pos] = rationalizer.mean_important_score
            
            print(rationalizer.mean_important_score)
            print(f'{target_pos + 1} / {generated_ids.shape[0]}')
            print(f'Target word     --> {tokenizer.decode(target_id)}', )
            print(f"Rational pos    --> {pos_rational}")
            print(f"Rational text   --> {text_rational}")
            print()
            
            raw_dict[tokenizer.decode(target_id)] = {'target_pos': target_pos.item(),
                                                    'Rational_pos': [ i.item() for i in pos_rational ],
                                                    'Rational_text': text_rational,
                                                    'importance_distribution': [ i.item() for i in rationalizer.mean_important_score]}
            
            # breakpoint()
        # breakpoint()
        # evaluation
        
        norm_suff_all = []
        norm_comp_all = []
        random_suff_all = []
        random_comp_all = []
        target_token_all = []

        table_details = [ ["target_pos", "target_token", "norm_suff", "random_suff", "norm_comp", "random_comp"] ]
        #random_importance_scores = normalise_random(torch.rand(importance_scores.size(), device=device))
        

        for target_pos in torch.arange(input_ids.shape[0], generated_ids.shape[0], args.stride):
            
            if target_pos in skipped_pos:
                continue

            target_token = tokenizer.decode(generated_ids[target_pos])
            target_token_all.append(target_token)

            input_ids_step = torch.unsqueeze(generated_ids[:target_pos], 0)
            target_id_step = torch.unsqueeze(generated_ids[target_pos], 0)
            importance_score_step = torch.unsqueeze(importance_score_map[target_pos - input_ids.shape[0], :target_pos], 0)
            random_importance_scores = normalise_random(torch.rand(importance_score_step.size(), device=device))

            norm_suff = soft_norm_suff_evaluator.evaluate(input_ids_step, target_id_step, importance_score_step)
            random_suff = soft_norm_suff_evaluator.evaluate(input_ids_step, target_id_step, random_importance_scores)
            norm_suff_all.append(norm_suff)
            random_suff_all.append(random_suff)

            norm_comp = soft_norm_comp_evaluator.evaluate(input_ids_step, target_id_step, importance_score_step)
            random_comp = soft_norm_comp_evaluator.evaluate(input_ids_step, target_id_step, random_importance_scores)
            norm_comp_all.append(norm_comp)
            random_comp_all.append(random_comp)

            table_details.append([target_pos.item() + 1, target_token, norm_suff.item(), random_suff.item(), norm_comp.item(), random_comp.item()])
            print(f"target_pos: {target_pos + 1}, target_token: {target_token}, norm_suff: {norm_suff}, random_suff: {random_suff}, norm_comp: {norm_comp}, random_comp:{random_comp}")


        print(table_details)
        print("".center(50, "-"))
        print(random_comp_all)
        print("".center(50, "-"))
        print(random_suff_all)

        if len(norm_suff_all) <= 0:
            print(f'[Warn] No results for {i}')

        norm_suff_mean = torch.mean(torch.tensor(norm_suff_all, device=device))
        norm_comp_mean = torch.mean(torch.tensor(norm_comp_all, device=device))

        random_suff_mean = torch.mean(torch.tensor(random_suff_all, device=device))
        random_comp_mean = torch.mean(torch.tensor(random_comp_all, device=device))

        print(f"norm_suff_mean: {norm_suff_mean}, norm_comp_mean: {norm_comp_mean}")
        print(f"random_suff_mean: {random_suff_mean}, random_comp_mean: {random_comp_mean}")

        # export generated_texts

        raw_dumps = json.dumps(raw_dict, indent=4)
        with open(os.path.join(output_dir, f'{i}_raw.json'), 'w')  as outfile:
            outfile.write(raw_dumps)

        with open(os.path.join(output_dir, f'{i}_output.txt'), 'w') as outfile:
            outfile.writelines(generated_texts)

        # export table
        table_mean = [
            [ "norm_suff_mean", "random_suff_mean", "norm_comp_mean", "random_comp_mean", "final suff", "final comp", "target_tokens" ],
            [ norm_suff_mean.item(), random_suff_mean.item(), norm_comp_mean.item(), random_comp_mean.item(), (norm_suff_mean / random_suff_mean).item(), (random_comp_mean /norm_suff_mean).item(), "$".join(target_token_all)],
        ]

        
        with open(os.path.join(output_dir, f'{i}_details.csv'), 'w', newline='') as csvfile:
            csvWriter = csv.writer(csvfile)
            csvWriter.writerows(table_details)
        
        with open(os.path.join(output_dir, f'{i}_mean.csv'), 'w', newline='') as csvfile:
            csvWriter = csv.writer(csvfile)
            csvWriter.writerows(table_mean)

        if args.if_image:    # export plot
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

    except Exception as e:
        print(f'[Warn] failed on {i}')
        # print(type(e))
        # print(e)
        import traceback
        traceback.print_exception(e)
        raise e

        # continue
    