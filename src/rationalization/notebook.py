from transformers import AutoModelForCausalLM, AutoTokenizer
from rationalizer.aggregate_rationalizer import AggregateRationalizer
from rationalizer.importance_score_evaluator.delta_prob import DeltaProbImportanceScoreEvaluator
from rationalizer.stopping_condition_evaluator.top_k import TopKStoppingConditionEvaluator
from rationalizer.token_replacement.token_replacer.uniform import UniformTokenReplacer
from rationalizer.token_replacement.token_sampler.postag import POSTagTokenSampler

device = "cuda"
model_name = "KoboldAI/OPT-6.7B-Erebus"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='../../cache')
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir='../../cache').to(device)






import torch
import seaborn
import matplotlib.pyplot as plt
# construct rationalizer

rational_size = 5
rational_size_ratio = None
max_steps = 3000
replace_ratio_for_update = 0.3
topk_for_stopping=10
batch=5

token_sampler = POSTagTokenSampler(tokenizer=tokenizer, device=device)

stopping_condition_evaluator = TopKStoppingConditionEvaluator(
    model=model, 
    token_sampler=token_sampler, 
    top_k=topk_for_stopping, 
    top_n=rational_size, 
    top_n_ratio=rational_size_ratio, 
    tokenizer=tokenizer
)

importance_score_evaluator = DeltaProbImportanceScoreEvaluator(
    model=model, 
    tokenizer=tokenizer, 
    token_replacer=UniformTokenReplacer(
        token_sampler=token_sampler, 
        ratio=replace_ratio_for_update
    ),
    stopping_condition_evaluator=stopping_condition_evaluator,
    max_steps=max_steps
)

rationalizer = AggregateRationalizer(
    importance_score_evaluator=importance_score_evaluator,
    batch_size=batch,
    overlap_threshold=2,
    overlap_strict_pos=True,
    top_n=rational_size, 
    top_n_ratio=rational_size_ratio
)




input_string = "Model explanation is a difficult "
max_length = 10

# generate prediction 
input_ids = tokenizer(input_string, return_tensors='pt')['input_ids'][0].to(model.device)
generated_ids = model.generate(input_ids=torch.unsqueeze(input_ids, 0), max_length=max_length, do_sample=False)[0]
generated_texts = [ tokenizer.decode(token) for token in generated_ids ]
print(f'generated full sequence --> {generated_texts}')


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
    print(f'Target word     --> {tokenizer.decode(target_id)[0]}', )
    print(f"Rational pos    --> {pos_rational}")
    print(f"Rational text   --> {text_rational}")

    print()


    
seaborn.set(rc={ 'figure.figsize': (30, 10) })
s = seaborn.heatmap(
    importance_score_map, 
    xticklabels=generated_texts[:-1], 
    yticklabels=generated_texts[input_ids.shape[0]:], 
    annot=True, 
    square=True).set(title=f'Explanation for ""{input_string}"", for Model {model_name}')
s.set_xlabel('Importance distribution')
s.set_ylabel('Target')
scatter_fig = s.get_figure()
scatter_fig.savefig('visual/t1.png')
print(' done')
