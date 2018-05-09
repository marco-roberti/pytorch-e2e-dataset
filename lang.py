import re
import unicodedata
from abc import ABC, abstractmethod
from typing import List

SOS_token = 0
EOS_token = 1


def _unicode_to_ascii(s: str):
    """Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427"""
    return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
    )


def _split_sentence(sentence: str) -> List[str]:
    sentence = re.sub(r'([^A-Za-z ])', r' \1 ', sentence)
    words = filter(lambda s: s != '', sentence.split(' '))
    return list(words)


class AbstractVocabulary(ABC):
    def __init__(self):
        self.token2index = {}
        self.token2count = {}
        self.index2token = {0: "<SOS>", 1: "<EOS>"}

    def __len__(self):
        return len(self.index2token)

    def _add_token(self, token: str) -> int:
        if token not in self.token2index:
            self.token2index[token] = len(self)
            self.token2count[token] = 1
            self.index2token[len(self)] = token
        else:
            self.token2count[token] += 1
        return self.token2index[token]

    @abstractmethod
    def add_sentence(self, sentence: str) -> List[int]:
        """Return the encoded sentence"""
        pass

    @abstractmethod
    def to_string(self, sequence: List[int]) -> str:
        pass


class WordVocabulary(AbstractVocabulary):

    def add_sentence(self, sentence: str) -> List[int]:
        sentence = _unicode_to_ascii(sentence)
        words = _split_sentence(sentence)
        sentence_enc = []
        for word in words:
            sentence_enc.append(self._add_token(word))
        sentence_enc.append(EOS_token)
        return sentence_enc

    def to_string(self, sequence: List[int]) -> str:
        return ' '.join([self.index2token[i] for i in sequence])


class CharVocabulary(AbstractVocabulary):
    def add_sentence(self, sentence: str) -> List[int]:
        sentence = _unicode_to_ascii(sentence)
        sentence_enc = []
        for c in sentence:
            sentence_enc.append(self._add_token(c))
        sentence_enc.append(EOS_token)
        return sentence_enc

    def to_string(self, sequence: List[int]) -> str:
        return ''.join([self.index2token[i] for i in sequence])
