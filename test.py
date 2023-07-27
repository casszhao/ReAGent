
from transformers import AutoModelForCausalLM, AutoModelWithLMHead, AutoTokenizer
import os


input_string = "sklm happy datys"

model = AutoModelWithLMHead.from_pretrained("gpt2-medium").to('cuda')
tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")

input_ids = tokenizer(input_string, return_tensors='pt')['input_ids'].to(model.device)
generated_input = model.generate(input_ids=input_ids, do_sample=False) 
generated_logits = model(input_ids=input_ids)['logits']


print(' ---- small ')
print(generated_logits)
