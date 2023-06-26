from rationalization.greedy_masking.huggingface import rationalize_lm
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForSeq2SeqLM

# Load model from Hugging Face
# model = AutoModelWithLMHead.from_pretrained("keyonvafa/compatible-gpt2")
# tokenizer = AutoTokenizer.from_pretrained("keyonvafa/compatible-gpt2")
model = AutoModelForSeq2SeqLM.from_pretrained("gpt2-medium")
tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
# model.cuda()
model.eval()

# Generate sequence
print("".center(50, "-"))
input_string = "I love eating breakfast out the"
input_ids = tokenizer(input_string, return_tensors='pt')['input_ids'].to(model.device)
generated_input = model.generate(input_ids=input_ids, pad_token_id=50256, num_return_sequences=1, do_sample=False)[0] 
print(' generated input -->', tokenizer.decode(generated_input))
# Rationalize sequence with greedy rationalization
# rationales, rationalization_log = rationalize_lm(model, generated_input, tokenizer, verbose=True)

print("".center(50, "-"))
input_string = "She loves eating breakfast in the"
input_ids = tokenizer(input_string, return_tensors='pt')['input_ids'].to(model.device)
generated_input = model.generate(input_ids=input_ids, max_length=8, do_sample=False)[0] 
print(' generated input -->', tokenizer.decode(generated_input))

print("".center(50, "-"))
input_string = "I loves cooking lunch in the"
input_ids = tokenizer(input_string, return_tensors='pt')['input_ids'].to(model.device)
generated_input = model.generate(input_ids=input_ids, max_length=8, do_sample=False)[0] 
print(' generated input -->', tokenizer.decode(generated_input))