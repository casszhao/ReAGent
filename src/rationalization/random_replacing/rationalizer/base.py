from .importance_score.evaluator import ImportanceScoreEvaluator
from .utils.traceable import Traceable


class BaseRationalizer(Traceable):

    def __init__(self, importance_score_evaluator: ImportanceScoreEvaluator) -> None:
        super().__init__()

        self.importance_score_evaluator = importance_score_evaluator
