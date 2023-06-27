from rationalization.greedy_masking.huggingface import rationalize_lm
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForSeq2SeqLM
from datasets import load_dataset

# books = load_dataset("opus_books", "en-fr")
# books = books["train"].train_test_split(test_size=0.2)
checkpoint = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
# books["train"][0]
text = "translate English to Chinese: testing ."
length = len(text.split(' ')) * 2

source_lang = "en"
target_lang = "zh"
prefix = "translate English to Chinese: "

text = prefix + text

inputs = tokenizer(text, return_tensors="pt").input_ids
outputs = model.generate(inputs, max_new_tokens=length, do_sample=True, top_k=30, top_p=0.95)
text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"==>> text_output: {text_output}")
print(f"==>> outputs: {outputs}")


# def preprocess_function(examples):
#     inputs = [prefix + example[source_lang] for example in examples["translation"][source_lang]]
#     targets = [example[target_lang] for example in examples["translation"][target_lang]]
#     model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
#     return model_inputs


# def postprocess_text(preds, labels):
#     preds = [pred.strip() for pred in preds]
#     labels = [[label.strip()] for label in labels]

#     return preds, labels


