
from fairseq.models.transformer import TransformerModel

class FairseqTokenizerWrapper:

    def __init__(self, fairseq_model: TransformerModel) -> None:
        self.fairseq_model = fairseq_model

    def decode(self, tokens: list[int], source: bool=False) -> str:
        return self.decode_sequence(tokens, source)

    def encode(self, text: str, source: bool=False) -> list[int]:
        return self.encode_sequence(text, source)
    
    def decode_single_word(self, word_index: int, source: bool=False) -> str:
        if source:
            dictionary = self.fairseq_model.task.source_dictionary
        else:
            dictionary = self.fairseq_model.task.target_dictionary
        if self.fairseq_model.bpe is None:
            return dictionary.symbols[word_index]
        return self.fairseq_model.bpe.decode(dictionary.string([word_index]))

    def decode_sequence(self, tokens: list[int], source: bool=False) -> str:
        return " ".join([self.decode_single_word(token, source) for token in tokens])

    def encode_single_word(self, word: str, source: bool=False) -> int:
        if source:
            dictionary = self.fairseq_model.task.source_dictionary
        else:
            dictionary = self.fairseq_model.task.target_dictionary
        if self.fairseq_model.bpe is None:
            return dictionary.indices[word]

        # TODO: 
        raise NotImplementedError("TBC")
        # return self.fairseq_model.bpe.encode(dictionary.string([word_index]))

    def encode_sequence(self, text: str, source: bool=False) -> list[int]:
        words = text.split(" ")
        return [self.encode_single_word(word, source) for word in words]
