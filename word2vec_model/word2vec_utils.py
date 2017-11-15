import nltk
import numpy as np

import sklearn
import math
from stop_words import get_stop_words


def vectors_cosine_similarity(vector1, vector2):
    similarity = sklearn.metrics.pairwise.cosine_similarity(vector1.reshape(1, -1), vector2.reshape(1, -1))
    return round(similarity[0][0], 3)


def tokenize_to_lower_case(sentence):
    tokenize = nltk.word_tokenize(sentence)
    return [t.lower() for t in tokenize]


def sentence2vectors(sentence, word2vec_model, to_lower_case=False, exclude_stopwords=False):
    stop_words = get_stop_words('en')
    vectors_by_tokens = {}
    sentence_tokens = nltk.word_tokenize(sentence)

    for sentence_token in sentence_tokens:
        if to_lower_case == True:
            sentence_token = sentence_token.lower()
        if exclude_stopwords == True:
            if sentence_token in stop_words:
                continue
        try:
            vectors_by_tokens[sentence_token] = word2vec_model.wv[sentence_token]
        except Exception as E:
            print(E)
            continue
    return vectors_by_tokens


def sentence_vectors_mean(vectors_by_tokens):
    if len(vectors_by_tokens) == 0:
        print("WARN no vectors by tokens")
        return []
    try:
        inited = False
        result = None
        for key, value in vectors_by_tokens.items():
            if not inited:
                result = np.zeros(value.shape, dtype=np.float64)
                inited = True
            result += np.asarray(value)
        return result / len(vectors_by_tokens)
    except Exception as E:
        print(E)
