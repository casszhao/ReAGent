import torch


class StoppingConditionEvaluator():
    """Base class for Stopping Condition Evaluators
    
    """
    
    def __init__(self):
        """Dummy Constructor
        
        """
        raise NotImplementedError()

    def evaluate(self, input_ids: torch.Tensor, target_id: torch.Tensor, importance_score: torch.Tensor) -> bool:
        """Dummy evaluate
        
        """
        raise NotImplementedError()
