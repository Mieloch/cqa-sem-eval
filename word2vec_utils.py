import nltk
import numpy as np

import sklearn
import math

def vectors_cosine_similarity(vector1, vector2):
    similarity = sklearn.metrics.pairwise.cosine_similarity(vector1.reshape(1, -1), vector2.reshape(1, -1))
    return round(similarity[0][0],3)


def tokenize_to_lower_case(sentence):
    tokenize = nltk.word_tokenize(sentence)
    return [t.lower() for t in tokenize]


def sentence2vectors(sentence, word2vec_model, lower_case=False):
    vectors_by_tokens = {}
    sentence_tokens = nltk.word_tokenize(sentence)
    if lower_case:
        sentence_tokens = [t.lower() for t in sentence_tokens]
    for token in sentence_tokens:
        vectors_by_tokens[token] = word2vec_model.wv[token]
    return vectors_by_tokens


def sentence_vectors_mean(vectors_by_tokens):
    try:
        inited = False
        result = None
        for key, value in vectors_by_tokens.items():
            if not inited:
                result = np.zeros(value.shape, dtype=np.float64)
                inited = True
            result += np.asarray(value)
        return result / len(vectors_by_tokens)
    except:
        print("lipa")
