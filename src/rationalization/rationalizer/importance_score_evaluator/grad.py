import logging

import torch
from transformers import AutoModelWithLMHead, AutoTokenizer
from .base import BaseImportanceScoreEvaluator
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

class GradientImportanceScoreEvaluator(BaseImportanceScoreEvaluator):
    """Importance Score Evaluator
    
    """

    def __init__(self, model: AutoModelWithLMHead, tokenizer: AutoTokenizer, grad_type: str) -> None:
        """Constructor

        Args:
            model: A Huggingface AutoModelWithLMHead model
            tokenizer: A Huggingface AutoTokenizer
            grad_type: grad_type in (integrated, norm, signed)

        """

        super().__init__(model, tokenizer)

        self.grad_type = grad_type

    @torch.enable_grad()
    def evaluate(self, input_ids: torch.Tensor, target_id: torch.Tensor) -> torch.Tensor:
        """Evaluate importance score of input sequence

        Args:
            input_ids: input sequence [batch, sequence]
            target_id: target token [batch]

        Return:
            importance_score: evaluated importance score for each token in the input [batch, sequence]

        """

        # TODO: Only work when batch size is 1
        assert input_ids.shape[0] == 1, "Batch input not supported"
        assert target_id.shape[0] == input_ids.shape[0], "Inconsistent batch size"

        position_ids = torch.arange(len(input_ids[0]))[None].long().cuda()

        if isinstance(self.model, GPT2LMHeadModel):
            gpt2Model: GPT2LMHeadModel = self.model
            word_token_embeds = gpt2Model.transformer.wte(input_ids)
            position_embeds = gpt2Model.transformer.wpe(position_ids)
        elif isinstance(self.model, OPTForCausalLM):
            optModel: OPTForCausalLM = self.model
            word_token_embeds = optModel.model.decoder.embed_tokens(input_ids)
            position_embeds = optModel.model.decoder.embed_positions(position_ids)
        else:
            raise ValueError(f"Unsupported model {type(self.model)}")

        pos_encoded_embeddings = word_token_embeds + position_embeds

        if self.grad_type == 'integrated':
            # Approximate integrated gradient.
            path_integral_steps = 100
            all_gradients = []
            for i in range(0, path_integral_steps + 1):
                path_initial_embeds = word_token_embeds * (i / path_integral_steps)
                path_logits = self.model(inputs_embeds=path_initial_embeds)['logits']
                path_target_probs = path_logits[:, -1].log_softmax(-1)[0, target_id]
                gradients = torch.autograd.grad(
                    path_target_probs, 
                    path_initial_embeds, 
                    retain_graph=True)[0].detach()
                all_gradients.append(gradients)
            path_integral = torch.sum(torch.cat(all_gradients), 0)
            integrated_gradient = torch.sum(
                path_integral[None] / 
                (path_integral_steps + 1) * word_token_embeds, -1)[0]
            logit_importance_score = torch.unsqueeze(integrated_gradient, 0)
        else:
            full_logits = self.model(inputs_embeds=word_token_embeds)['logits']
            embedding_grad = torch.autograd.grad(
                full_logits[0, -1].log_softmax(-1)[target_id], word_token_embeds)[0]
            if self.grad_type == 'norm':
                grad_scores = embedding_grad.norm(dim=-1).view(-1)
            elif self.grad_type == 'signed':
                grad_scores = torch.sum(embedding_grad * pos_encoded_embeddings, dim=-1)[0]
            logit_importance_score = torch.unsqueeze(grad_scores, 0)
        
        self.important_score = torch.softmax(logit_importance_score, -1)
        return self.important_score
