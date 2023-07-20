
from transformers import AutoModelForCausalLM, AutoModelWithLMHead, AutoTokenizer
import os
os.environ['TRANSFORMERS_CACHE'] = '/mnt/parscratch/users/cass/cache/'

# facebook/opt-350m
# name 


input_string = "sklm happy datys"

model = AutoModelWithLMHead.from_pretrained("gpt2-medium").to('cuda')
tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")

input_ids = tokenizer(input_string, return_tensors='pt')['input_ids'].to(model.device)
generated_input = model.generate(input_ids=input_ids, do_sample=False) 
generated_logits = model(input_ids=input_ids)['logits']


print(' ---- small ')
print(generated_logits)


model = AutoModelForCausalLM.from_pretrained("bigscience/bloom", cache_dir='/mnt/parscratch/users/cass/cache/').to('cuda')
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom", cache_dir='/mnt/parscratch/users/cass/cache/')

input_ids = tokenizer(input_string, return_tensors='pt')['input_ids'].to(model.device)
generated_input = model(input_ids=input_ids) 

generated_logits = model(input_ids=input_ids)['logits']
print(' ---- big ')
print(generated_logits)