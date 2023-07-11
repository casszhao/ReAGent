import logging

import torch
from transformers import AutoModelWithLMHead, AutoTokenizer
from .base import BaseImportanceScoreEvaluator

class AttentionImportanceScoreEvaluator(BaseImportanceScoreEvaluator):
    """Importance Score Evaluator
    
    """

    def __init__(self, model: AutoModelWithLMHead, tokenizer: AutoTokenizer, attn_type: str) -> None:
        """Constructor

        Args:
            model: A Huggingface AutoModelWithLMHead model
            tokenizer: A Huggingface AutoTokenizer
            method: method

        """

        super().__init__(model, tokenizer)

        self.attn_type = attn_type

    def evaluate(self, input_ids: torch.Tensor, target_id: torch.Tensor) -> torch.Tensor:
        """Evaluate importance score of input sequence

        Args:
            input_ids: input sequence [batch, sequence]
            target_id: target token [batch]

        Return:
            importance_score: evaluated importance score for each token in the input [batch, sequence]

        """

        outputs = self.model(input_ids, output_attentions=True)
        all_attentions = outputs['attentions']
        # Turn tuple into list, take mean over number of heads
        attentions = torch.mean(torch.stack(all_attentions), [2])
        num_layers, batch_size, seq_len, _ = attentions.shape

        if self.attn_type == 'last':
            # last layer, full batch, last token
            last_attention = attentions[-1, :, -1]
            logit_importance_score = last_attention
        elif self.attn_type == 'all':
            # Average over (all attention heads over) all layers of the last step.
            all_attentions = torch.mean(attentions, 0)[:, -1]
            logit_importance_score = all_attentions
        elif self.attn_type == 'rollout':
            residualized_attentions = (
                0.5 * attentions + 0.5 * torch.eye(seq_len)[None].to(attentions))
            rollout = residualized_attentions[0]
            for layer in range(1, num_layers):
                rollout = torch.matmul(residualized_attentions[layer], rollout)
            
            logit_importance_score = rollout[:, -1]
        else:
            raise ValueError(f"Invalid attn_type {self.attn_type}")

        self.important_score = torch.softmax(logit_importance_score, -1)
        return self.important_score
