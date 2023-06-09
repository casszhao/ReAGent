import json
import torch


def serialize_rational(
    filename: str,
    id: int,
    token_inputs: torch.Tensor,
    token_target: torch.Tensor,
    position_rational: torch.Tensor,
    tokenizer,
    important_score: torch.Tensor,
    comments: dict = None,
    compact: bool = False,
):
    data = {
        "$schema": "../docs/rational.schema.json",
        "id": id,
        "input-text": [tokenizer.decode([i]) for i in token_inputs],
        "input-tokens": [i.item() for i in token_inputs],
        "target-text": tokenizer.decode([token_target]),
        "target-token": token_target.item(),
        "importance-scores": [i.item() for i in important_score],
        "rational-size": position_rational.shape[0],
        "rational-positions": [i.item() for i in position_rational],
        "rational-text": [tokenizer.decode([i]) for i in token_inputs[position_rational]],
        "rational-tokens": [i.item() for i in token_inputs[position_rational]],
    }

    if comments:
        data["comments"] = comments

    indent = None if compact else 4
    json_str = json.dumps(data, indent=indent)

    with open(filename, 'w') as f_output:
        f_output.write(json_str)
