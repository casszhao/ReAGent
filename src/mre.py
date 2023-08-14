import inseq
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")

attribution_model = inseq.load_model("gpt2-medium", "attention")

input_tokens = tokenizer.encode("Test\n")         # [14402, 198]
predicted_next = tokenizer.encode("\n")           # [198]
target_tokens = input_tokens + predicted_next     # [14402, 198, 198]

# These will be passed to inseq
input_text = tokenizer.decode(input_tokens)       # 'Test\n'
target_text = tokenizer.decode(target_tokens)     # 'Test\n\n'

# e.g.
# out = attribution_model.attribute(
#   [ input_text ],
#   [ target_text ]
# )

# This happend inside inseq: re-encoded tokens wont match
input_tokens_re_encoded = tokenizer.encode(input_text)  # [14402, 198]
target_tokens_re_encoded = tokenizer.encode(target_text)  # [14402, 628]


# core issue
tokenizer.encode(tokenizer.decode([198, 198]))  # [628]
