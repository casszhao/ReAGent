
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name='KoboldAI/OPT-6.7B-Erebus'




tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='cache')
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir='cache')
model.to('cuda')



string = "When my flight landed in China, I converted my currency and slowly fell asleep. (I had a terrifying dream about my grandmother, but that’s a story for another time). I was staying in the capital city, "
full_ids = tokenizer(string, return_tensors="pt")["input_ids"].cuda()

model_prediction_full = tokenizer.decode(
                        model.generate(full_ids[:, :-1], max_length=full_ids.shape[1]+1, do_sample=False)[0])

print(model_prediction_full)



string = "When my flight landed in China, I converted my currency and slowly fell asleep. (I had a terrifying dream about my grandmother, but that’s a story for another time). I was staying in the capital city, Beijing."
full_ids = tokenizer(string, return_tensors="pt")["input_ids"].cuda()

model_prediction_full = tokenizer.decode(
                        model.generate(full_ids[:, :-1], max_length=full_ids.shape[1], do_sample=False)[0])

print(model_prediction_full)



string = "When my flight landed in China, I converted my currency and slowly fell asleep. (I had a terrifying dream about my grandmother, but that’s a story for another time). I was staying in the capital, "
full_ids = tokenizer(string, return_tensors="pt")["input_ids"].cuda()

model_prediction_full = tokenizer.decode(
                        model.generate(full_ids[:, :-1], max_length=full_ids.shape[1]+2, do_sample=False)[0])

print(model_prediction_full)

model_prediction_full = tokenizer.decode(
                        model.generate(full_ids[:, :-1], max_length=full_ids.shape[1]+1, do_sample=False)[0])

print(model_prediction_full)

