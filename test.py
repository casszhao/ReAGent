
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