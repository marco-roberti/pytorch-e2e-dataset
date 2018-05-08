import re
import unicodedata
from typing import List

SOS_token = 0
EOS_token = 1


class WordVocabulary:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def _add_word(self, word: str):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
        return self.word2index[word]

    def _split_sentence(self, sentence: str) -> List[str]:
        sentence = self._unicode_to_ascii(sentence)
        sentence = re.sub(r'([^A-Za-z ])', r' \1 ', sentence)
        words = filter(lambda s: s != '', sentence.strip().split(' '))
        return list(words)

    @staticmethod
    def _unicode_to_ascii(s: str):
        """Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427"""
        return ''.join(
                c for c in unicodedata.normalize('NFD', s)
                if unicodedata.category(c) != 'Mn'
        )

    def add_sentence(self, sentence: str) -> List[int]:
        words = self._split_sentence(sentence)
        sentence_enc = []
        for word in words:
            sentence_enc.append(self._add_word(word))
        return sentence_enc

    def to_string(self, sequence: List[int]) -> str:
        return ' '.join([self.index2word[i] for i in sequence])
