import numpy as np


class ForwardMaxMatch:
    """Simple tokenizer"""

    def __init__(self, dict_path):
        self.dict_path = dict_path
        self.word_dict = {}
        self.generate_dict()
        #with open(dict_path, 'r', encoding='utf-8') as f:
        #    for line in f:
        #        self.word_dict.add(line.strip())

    def generate_dict(self):
        """Generate dict"""
        with open(self.dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                items = line.strip().split()
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
        return words

