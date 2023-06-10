from transformers import AutoTokenizer
from token_replacement.token_sampler.postag import POSTagTokenSampler

if __name__ == "__main__":
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")

    input_string = "I love eating breakfast in the"
    input_ids = tokenizer(input_string, return_tensors='pt')['input_ids'].to("cuda")

    ts = POSTagTokenSampler(tokenizer, input_ids.device)

    for i in range(20):
        sampled_ids = ts.sample(input_ids)
        print([tokenizer.decode([i]) for i in sampled_ids[0]])
