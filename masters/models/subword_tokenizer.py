from collections import deque
from tqdm import tqdm
from scipy.stats import lognorm
import numpy as np
import pickle
import random


class SubwordTokenizer():
    def __init__(self, 
                corpus, 
                batch_size: int,
                top_k: int,
                token_length: int,
                mu: float,
                sigma2: float
                ):
        self.corpus = corpus
        self.batch_size = batch_size
        self.top_k = top_k
        self.token_length = token_length
        self.mu = mu
        self.sigma2 = sigma2
        self.vocab = []

    def build(self):
        train_queue = deque()
        substrings_freq = {}

        for start in tqdm(range(0, len(self.corpus), self.batch_size)):
            if len(train_queue) > 0:
                train_queue.popleft()

            batch = self.corpus.iloc[start:start+self.batch_size]
            train_queue.append(batch)

            for batch in train_queue:
                for word in batch:
                    subs = self._substrings(word, self.token_length)
                    for sub in subs:
                        substrings_freq[sub] = substrings_freq.get(sub, 0) + 1

        total_freq = sum(substrings_freq.values())
        for sub in substrings_freq:
            substrings_freq[sub] /= total_freq

        z = lognorm.rvs(self.sigma2, scale=np.exp(self.mu))

        scores = {}
        for sub, freq in substrings_freq.items():
            scores[sub] = freq + random.gauss(0, z)

        sorted_substrings = sorted(scores.items(), key=lambda x: x[1], reverse=False)
        self.vocab = [sub[0] for sub in sorted_substrings[:self.top_k]]

        return self
    
    def tokenize(self, word):
        tokens = []
        word_length = len(word)

        while word_length > 0:
            for i in range(word_length, 0, -1):
                sub = word[:i]
                if sub in self.vocab:
                    tokens.append(sub)
                    word = word[1:]
                    word_length = len(word)
                    break
                    
                else:
                    tokens.append(word[0])
                    word = word[1:]
                    word_length = len(word)

        return tokens

    def save_to_pickle(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"Tokenizer saved to {filename}")

    @staticmethod
    def load_from_pickle(filename: str):
        with open(filename, 'rb') as f:
            tokenizer = pickle.load(f)
        print(f"Tokenizer loaded from {filename}")
        return tokenizer
        
    @staticmethod
    def _substrings(word, length):
        return [word[i:i+length] for i in range(len(word) - length + 1)]