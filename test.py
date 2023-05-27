from huggingface.rationalization import rationalize_lm
from transformers import AutoTokenizer, AutoModelWithLMHead

# Load model from Hugging Face
model = AutoModelWithLMHead.from_pretrained("keyonvafa/compatible-gpt2")
tokenizer = AutoTokenizer.from_pretrained("keyonvafa/compatible-gpt2")
model.cuda()
model.eval()

# Generate sequence
input_string = "The Supreme Court on Tuesday"
input_ids = tokenizer(input_string, return_tensors='pt')['input_ids'].to(model.device)
generated_input = model.generate(input_ids=input_ids, max_length=16, do_sample=False)[0]
  
# Rationalize sequence with greedy rationalization
rationales, rationalization_log = rationalize_lm(model, generated_input, tokenizer, verbose=True)
