import logging
from typing import List

import numpy as np


class ForwardMaxMatch:
    """Simple tokenizer"""

    def __init__(self, dict_path):
        self.dict_path = dict_path
        self.word_dict = {}
        self.vec_size = None
        self.generate_dict()
        self.sentence_size = 0 # Maximum number of words in a sentence
        #with open(dict_path, 'r', encoding='utf-8') as f:
        #    for line in f:
        #        self.word_dict.add(line.strip())

    def generate_dict(self):
        """Generate dict"""
        line_idx = 0
        with open(self.dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_idx += 1
                if line_idx == 1:
                    continue
                items = line.strip().split()
                if not self.vec_size:
                    self.vec_size = len(items) - 1
                assert len(items) - 1 == self.vec_size
                word = items[0]
                vec = [float(e) for e in items[1:]]
                self.word_dict[word] = vec


    def cut(self, sentence: str):
        """
        1. Tokenization
        2. To vector
        """
        words = []
        i = 0
        while i < len(sentence):
            # Max length is 5
            for j in range(min(len(sentence) - i, 5), 0, -1):
                # Search from max size, util find word in dict or length=1
                if sentence[i:i + j] in self.word_dict or j == 1:
                    words.append(sentence[i:i + j])
                    i += j
                    break
        self.sentence_size = max(self.sentence_size, len(words))
        return words

    def to_one_dim_vec(self, words: List[str]):
        """
        Transform words to vector, simply average each word's vector
        Args:
            words: List of tokens in a sentence
        """
        if len(words) == 0:
            raise ValueError("Empty words list")
        if not self.vec_size:
            raise ValueError("Unknown vector size")
        vec = np.zeros(self.vec_size)
        for w in words:
            try:
                vec += np.array(self.word_dict[w])
            except KeyError as e:
                logging.info(f"Error getting vector for word: {w}, error: {e}")
                continue
        return np.divide(vec, len(words))

    def to_two_dim_vec(self, words: List[str]):
        """
        Transform words to 2d-vector, concat each word's embedding, padding to sentence size
        Args:
            words: List of tokens in a sentence
        """
        if len(words) == 0:
            raise ValueError("Empty words list")
        if not self.vec_size:
            raise ValueError("Unknown vector size")
        vec = np.zeros((self.vec_size, self.sentence_size))
        for i, w in enumerate(words):
            try:
                vec[:,i] = np.array(self.word_dict[w])
            except KeyError as e:
                logging.info(f"Error getting vector for word: {w}, error: {e}")
                continue
        return vec

    def to_two_dim_vec_v2(self, words: List[str]):
        """
        Transform words to 2d-vector, concat each word's embedding, padding to sentence size
        Args:
            words: List of tokens in a sentence
        """
        if len(words) == 0:
            raise ValueError("Empty words list")
        if not self.vec_size:
            raise ValueError("Unknown vector size")
        vec = np.zeros((self.vec_size, self.sentence_size))
        for i, w in enumerate(words):
            try:
                vec[:,i] = np.array(self.word_dict[w])
            except KeyError as e:
                logging.info(f"Error getting vector for word: {w}, error: {e}")
                continue
        return vec
