# project.py


import pandas as pd
import numpy as np
from pathlib import Path
import re
import requests
import time


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------

'''
start \x02 (beginning of text)
end \x03 (end of text)
'''
def get_book(url):
    request = requests.get(url)
    text = request.text
    pause = 0.5
    check = False
    time.sleep(0.5)

    book_content = ''
    start = '*** START'
    end = '*** END'

    for line in text.split('\n'):
        if start in line:
            check = True
            book_content+='\n'
        elif end in line:
            break
        elif check:
            book_content+=line + '\n'

    replace_content = book_content.replace('\r\n','\n')

    return replace_content
# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def tokenize(book_string):
    pattern = r'\b\w+\b|\b\d+\b|[^\s\w]'
    token = re.findall(pattern,book_string)
    token = ['\x02'] + token + ['\x03']
    return token


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


class UniformLM(object):


    def __init__(self, tokens):
        self.mdl = self.train(tokens)
    def train(self, tokens):
        unique_tokens = set(tokens)
        probability = 1/ len(unique_tokens)
        output = pd.Series(probability,index = unique_tokens)
        return output
    def probability(self, words):
        for i in self.mdl.index:
            if  i in words:
                return (1 / len(self.mdl))**len(words)
        return 0.0
    def sample(self, M):
        sequence = np.random.choice(self.mdl.index,M)
        return " ".join(sequence)


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


class UnigramLM(object):
    
    def __init__(self, tokens):

        self.mdl = self.train(tokens)
    
    def train(self, tokens):
        
        return pd.Series(tokens).value_counts(normalize=True)
    
    def probability(self, words):
        if any( i not in self.mdl.index for i in words):
            return 0
        return self.mdl[words].prod()
            
        
    def sample(self, M):
        sequence = np.random.choice(self.mdl.index, size = M, p = self.mdl.values)
        
        return " ".join(sequence)
        


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------

class NGramLM(object):
    
    def __init__(self, N, tokens):
        self.N = N
        self.ngrams = self.create_ngrams(tokens)
        self.mdl = self.train()

        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            self.prev_mdl = NGramLM(N-1, tokens)

    def create_ngrams(self, tokens):
        n = self.N
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    def train(self):
        ngram_count = {}
        n1gram_count = {}

        for ngram in self.ngrams:
            ngram_count[ngram] = ngram_count.get(ngram, 0) + 1
            n1gram = ngram[:-1]
            n1gram_count[n1gram] = n1gram_count.get(n1gram, 0) + 1

        ngrams_data = []
        for ngram, count in ngram_count.items():
            n1gram = ngram[:-1]
            prob = count / n1gram_count[n1gram]
            ngrams_data.append((ngram, n1gram, prob))

        return pd.DataFrame(ngrams_data, columns=['ngram', 'n1gram', 'prob'])

    def probability(self, words):
        if len(words) < self.N:
            return self.prev_mdl.probability(words)

        ngram_set = set(tuple(words[i:i+self.N]) for i in range(len(words) - self.N + 1))

        filtered = self.mdl[self.mdl["ngram"].isin(ngram_set)]
        prob = filtered.groupby('n1gram')['prob'].prod().prod()

        r = words[:self.N-1]
        prob1 = self.prev_mdl.probability(r)

        return prob * prob1
    
    def create_indexed_ngrams(self):
        indexed_ngrams = {}
        for ngram in self.ngrams:
            prefix = tuple(ngram[:-1])
            last_token = ngram[-1]
            if prefix not in indexed_ngrams:
                indexed_ngrams[prefix] = []
            indexed_ngrams[prefix].append(last_token)
        return indexed_ngrams

    def sample(self, M):
        end_string = ["\x02"]
        current_prefix = ("\x02",)
        indexed_ngrams = self.create_indexed_ngrams()
        possible_tokens = indexed_ngrams.get(current_prefix, ["\x03"])

        while len(end_string) < M:
            next_token = np.random.choice(possible_tokens)
            end_string.append(next_token)
            current_prefix = current_prefix[1:] + (next_token,)
            possible_tokens = indexed_ngrams.get(current_prefix, ["\x03"])

        end_string.append("\x03")
        final_string = ' '.join(end_string)
        return final_string
