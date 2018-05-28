# coding=utf-8
import re
import unicodedata
from abc import ABC, abstractmethod
from typing import List

PAD_token = 0
SOS_token = 1
EOS_token = 2


class AbstractVocabulary(ABC):
    def __init__(self):
        self.token2index = {}
        self.token2count = {}
        self.index2token = {
            PAD_token: '<PAD>',
            SOS_token: '<SOS>',
            EOS_token: '<EOS>'
        }

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

    @abstractmethod
    def to_list(self, sequence: str) -> List[int]:
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

    def to_list(self, sentence: str) -> List[int]:
        sentence = _unicode_to_ascii(sentence)
        words = _split_sentence(sentence)
        return [self.token2index[w] for w in words]

    def to_string(self, sequence: List[int]) -> str:
        if EOS_token in sequence:
            eos_position = sequence.index(EOS_token)
        else:
            eos_position = len(sequence)
        return ' '.join([self.index2token[i] for i in sequence[:eos_position]])


class CharVocabulary(AbstractVocabulary):
    def add_sentence(self, sentence: str) -> List[int]:
        sentence = _unicode_to_ascii(sentence)
        sentence_enc = []
        for c in sentence:
            sentence_enc.append(self._add_token(c))
        sentence_enc.append(EOS_token)
        return sentence_enc

    def to_list(self, sentence: str) -> List[int]:
        sentence = _unicode_to_ascii(sentence)
        return [self.token2index[c] for c in sentence]

    def to_string(self, sequence: List[int]) -> str:
        if EOS_token in sequence:
            eos_position = sequence.index(EOS_token)
        else:
            eos_position = len(sequence)
        return ''.join([self.index2token[i] for i in sequence[:eos_position]])

    def __str__(self):
        return ''.join(self.token2index.keys())


def _split_sentence(sentence: str) -> List[str]:
    sentence = re.sub(r'([^A-Za-z ])', r' \1 ', sentence)
    words = filter(lambda s: s != '', sentence.split(' '))
    return list(words)


def _unicode_to_ascii(s: str):
    """Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427"""
    return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
    )
