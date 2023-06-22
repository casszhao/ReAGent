from rationalization.greedy_masking.huggingface import rationalize_lm
from transformers import AutoTokenizer, AutoModelWithLMHead

# Load model from Hugging Face
model = AutoModelWithLMHead.from_pretrained("keyonvafa/compatible-gpt2")
tokenizer = AutoTokenizer.from_pretrained("keyonvafa/compatible-gpt2")

model.cuda()
model.eval()

# Generate sequence
input_string = "I love eating breakfast in the"
input_ids = tokenizer(input_string, return_tensors='pt')['input_ids'].to(model.device)
generated_input = model.generate(input_ids=input_ids, max_length=8, do_sample=False)[0]
  
# Rationalize sequence with greedy rationalization
rationales, rationalization_log = rationalize_lm(model, generated_input, tokenizer, verbose=True)

print(' generated input -->', tokenizer.decode(generated_input))