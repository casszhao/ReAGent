from rationalization.greedy_masking.huggingface import rationalize_lm
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForSeq2SeqLM
from datasets import load_dataset
import evaluate

metric = evaluate.load("sacrebleu")

from transformers import FSMTForConditionalGeneration, FSMTTokenizer
mname = "facebook/wmt19-de-en"
tokenizer = FSMTTokenizer.from_pretrained(mname)
model = FSMTForConditionalGeneration.from_pretrained(mname)

input = "Maschinelles Lernen ist großartig, oder?"
input_ids = tokenizer.encode(input, return_tensors="pt")
outputs = model.generate(input_ids)
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded) # Machine learning is great, isn't it?

####### batch

dataset = load_dataset('stas/wmt14-en-de-pre-processed', split='test')
print(dataset[0])

avg_blue = 0
for i in range(999):
    lang_to_translate = [dataset[i]['translation']['de']]
    translated = [dataset[i]['translation']['en']]
    scores = metric.compute(predictions=lang_to_translate, references=translated)
    print("".center(50, "-"))
    print(f"==>> scores: {scores}")
    blue_scores = scores['score']
    print(f"==>> blue_scores: {blue_scores}")
    avg_blue += blue_scores

avg_blue = avg_blue/999
print(avg_blue)


