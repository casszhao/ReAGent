from ..token_sampler.base import TokenSampler

class TokenReplacer():
    def __init__(self, token_sampler: TokenSampler):
        self.token_sampler = token_sampler

    def sample(self, input):
        raise NotImplementedError()
