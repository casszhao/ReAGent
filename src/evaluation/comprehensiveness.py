class ComprehensivenessEvaluator():

    def __init__(self, model) -> None:
        self.model = model

    def evaluate(self, input_ids, target_id):
        # original prob
        logits_original = self.model(input_ids)["logits"]
        prob_original = logits_original[target_id]
        prob_target_original = prob_original[target_id]


        # masked prob


        # score