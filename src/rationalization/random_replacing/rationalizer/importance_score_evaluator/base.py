import logging

import torch
from transformers import AutoModelWithLMHead, AutoTokenizer
from typing_extensions import override

from ..stopping_condition_evaluator.base import StoppingConditionEvaluator
from ..token_replacement.token_replacer.base import TokenReplacer
from ..utils.traceable import Traceable


class BaseImportanceScoreEvaluator(Traceable):
    """Importance Score Evaluator
    
    """

    def __init__(self, model: AutoModelWithLMHead, tokenizer: AutoTokenizer, token_replacer: TokenReplacer, stopping_condition_evaluator: StoppingConditionEvaluator) -> None:
        """Base Constructor

        Args:
            model: A Huggingface AutoModelWithLMHead model
            tokenizer: A Huggingface AutoTokenizer
            token_replacer: A TokenReplacer
            stopping_condition_evaluator: A StoppingConditionEvaluator

        """

        self.model = model
        self.tokenizer = tokenizer
        self.token_replacer = token_replacer
        self.stopping_condition_evaluator = stopping_condition_evaluator
        self.important_score = None

        self.trace_importance_score = None
        self.trace_target_likelihood_original = None
        self.num_steps = 0

    def update_importance_score(self, logit_importance_score: torch.Tensor, input_ids: torch.Tensor, target_id: torch.Tensor, prob_original_target: torch.Tensor) -> torch.Tensor:
        """Update importance score by one step

        Args:
            logit_importance_score: Current importance score in logistic scale [batch]
            input_ids: input tensor [batch, sequence]
            target_id: target tensor [batch]
            prob_original_target: predictive probability of the target on the original sequence [batch]

        Return:
            logit_importance_score: updated importance score in logistic scale [batch]

        """
        raise NotImplementedError()

    def evaluate(self, input_ids: torch.Tensor, target_id: torch.Tensor) -> torch.Tensor:
        """Evaluate importance score of input sequence

        Args:
            input_ids: input sequence [batch, sequence]
            target_id: target token [batch]

        Return:
            importance_score: evaluated importance score for each token in the input [batch, sequence]

        """

        raise NotImplementedError()
    
    @override
    def trace_start(self):
        """Start tracing
        
        """
        super().trace_start()

        self.trace_importance_score = []
        self.trace_target_likelihood_original = -1
        self.stopping_condition_evaluator.trace_start()

    @override
    def trace_stop(self):
        """Stop tracing
        
        """
        super().trace_stop()

        self.trace_importance_score = None
        self.trace_target_likelihood_original = None
        self.stopping_condition_evaluator.trace_stop()
