import logging

import torch
from botorch import fit_fully_bayesian_model_nuts
from botorch.acquisition import qExpectedImprovement
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.optim import optimize_acqf
from transformers import AutoModelWithLMHead, AutoTokenizer

from ..stopping_condition_evaluator.base import StoppingConditionEvaluator
from ..token_replacement.token_replacer.ranking import RankingTokenReplacer
from .base import BaseImportanceScoreEvaluator


class BayesianOptimizationImportanceScoreEvaluator(BaseImportanceScoreEvaluator):
    """Importance Score Evaluator
    
    """

    def __init__(self, model: AutoModelWithLMHead, tokenizer: AutoTokenizer, token_replacer: RankingTokenReplacer, stopping_condition_evaluator: StoppingConditionEvaluator, 
                 sample_multiplier: float, sample_increment: int, training_config: dict, optimizing_config: dict) -> None:
        """Constructor

        Args:
            model: A Huggingface AutoModelWithLMHead model
            tokenizer: A Huggingface AutoTokenizer
            token_replacer: A TokenReplacer
            stopping_condition_evaluator: A StoppingConditionEvaluator

        """
        super().__init__(model, tokenizer, token_replacer, stopping_condition_evaluator)
        self.token_replacer = token_replacer
        self.sample_multiplier = sample_multiplier
        self.sample_increment = sample_increment
        self.training_config = training_config
        self.optimizing_config = optimizing_config

        self.important_score = None

        self.trace_importance_score = None
        self.trace_target_likelihood_original = None
        self.num_steps = 0

    def bayesian_opti(self, logit_importance_scores, delta_prob_targets):
        """SAASBO optimization for logit_importance_score

        Args:
            logit_importance_scores: Current importance score in logistic scale [samples]
            delta_prob_targets: input tensor [samples]

        Return:
            logit_importance_score: updated importance score in logistic scale [1]

        """

        # TODO: Use multiple candidates and add them into the sample set

        opti_target = -1 * delta_prob_targets  # Flip the sign since we want to minimize the delta
        # See: https://botorch.org/api/models.html#botorch.models.fully_bayesian.SaasFullyBayesianSingleTaskGP
        gp = SaasFullyBayesianSingleTaskGP(
            train_X=logit_importance_scores,
            train_Y=opti_target,
        )
        with torch.enable_grad():
            fit_fully_bayesian_model_nuts(
                gp,
                disable_progbar=True,
                **self.training_config
            )

        # Maybe?: https://botorch.org/api/acquisition.html#botorch.acquisition.monte_carlo.qExpectedImprovement
        ei = qExpectedImprovement(model=gp, best_f=opti_target.max())

        # Maybe?: https://botorch.org/api/optim.html#botorch.optim.optimize.optimize_acqf
        with torch.enable_grad():
            # TODO: make bounds config
            lower_bounds = torch.ones(1, logit_importance_scores.shape[1]) * -1000
            upper_bounds = torch.ones(1, logit_importance_scores.shape[1]) * 1000

            candidates, acq_values = optimize_acqf(
                ei,
                bounds=torch.cat([lower_bounds, upper_bounds]).to(opti_target.device),
                q=1,
                **self.optimizing_config
            )

        # pick candidates
        logit_importance_score = candidates[0]
        return logit_importance_score

    def expand_samples(self, input_ids, target_id, prob_original_target):
        num_expand = self.samples_logit_importance_score.shape[0] * (self.sample_multiplier - 1) + self.sample_increment
        logit_importance_score = torch.rand([num_expand, input_ids.shape[1]], device=input_ids.device)

        self.token_replacer.set_score(logit_importance_score)
        input_ids_replaced, mask_replaced = self.token_replacer.sample(input_ids)
        logits_replaced = self.model(input_ids_replaced)['logits']
        prob_replaced_target = torch.softmax(logits_replaced[:, input_ids.shape[1] - 1, :], -1)[:, target_id]

        delta_prob_target = prob_original_target - prob_replaced_target

        self.samples_logit_importance_score = torch.cat([self.samples_logit_importance_score, logit_importance_score])
        self.samples_delta_prob_target = torch.cat([self.samples_delta_prob_target, delta_prob_target])

        logging.debug(f"Expand sample set to {self.samples_delta_prob_target.shape[0]}")

    def evaluate(self, input_ids: torch.Tensor, target_id: torch.Tensor) -> torch.Tensor:
        """Evaluate importance score of input sequence

        Args:
            input_ids: input sequence [1, sequence]
            target_id: target token [1]

        Return:
            importance_score: evaluated importance score for each token in the input [1, sequence]

        """

        self.stop_mask = torch.zeros([1], dtype=torch.bool, device=input_ids.device)

        # Inference p^{(y)} = p(y_{t+1}|y_{1...t})

        logits_original = self.model(input_ids)['logits']
        prob_original_target = torch.softmax(logits_original[:, input_ids.shape[1] - 1, :], -1)[:, target_id]

        if self.trace_target_likelihood_original != None:
            self.trace_target_likelihood_original = prob_original_target

        # Initialize samples
        self.samples_logit_importance_score = torch.zeros([0, input_ids.shape[1]])
        self.samples_delta_prob_target = torch.zeros([0, 1])

        # TODO: limit max steps
        self.num_steps = 0
        while True:
            self.num_steps += 1
            
            # Update importance score
            self.expand_samples(input_ids, target_id, prob_original_target)
            logit_importance_score = self.bayesian_opti(self.samples_logit_importance_score, self.samples_delta_prob_target)

            self.important_score = torch.softmax(logit_importance_score, -1)
            if self.trace_importance_score != None:
                self.trace_importance_score.append(self.important_score)

            # Evaluate stop condition
            self.stop_mask = self.stopping_condition_evaluator.evaluate(input_ids, target_id, self.important_score)
            if torch.prod(self.stop_mask) > 0:
                break
        
        logging.info(f"Importance score evaluated in {self.num_steps} steps.")

        return self.important_score
