import nltk
import numpy as np
import gensim
import sklearn
import math
import os
from stop_words import get_stop_words


def load_word2vec_model(model_name):
    print("Loading word2vec model {}".format(model_name))
    path = os.path.dirname(os.path.abspath(__file__))
    model = None
    if model_name == "Q1_model" or model_name == "SemEval2016-Task3-CQA-QL-dev_model":
        model = gensim.models.Word2Vec.load(path + "/" + model_name)
    elif model_name == "GoogleNews-vectors-negative300.bin":
        model = gensim.models.KeyedVectors.load_word2vec_format(path + "/" + model_name,
                                                                binary=True)
    else:
        raise Exception("Unknown word2vec model {}".format(model_name))
    print("Loading word2vec model {} [DONE]".format(model_name))
    return model


def vectors_cosine_similarity(vector1, vector2):
    similarity = sklearn.metrics.pairwise.cosine_similarity(vector1.reshape(1, -1), vector2.reshape(1, -1))
    return round(similarity[0][0], 3)


def tokenize_to_lower_case(sentence):
    tokenize = nltk.word_tokenize(sentence)
    return [t.lower() for t in tokenize]


def sentence2vector_list(sentence, word2vec_model, to_lower_case=False, exclude_stopwords=False):
    stop_words = get_stop_words('en')
    vectors = []
    sentence_tokens = nltk.word_tokenize(sentence)

    for sentence_token in sentence_tokens:
        if to_lower_case == True:
            sentence_token = sentence_token.lower()
        if exclude_stopwords == True:
            if sentence_token in stop_words:
                continue
        try:
            vectors.append(word2vec_model.wv[sentence_token])
        except Exception as E:
            print(E)
            continue
    return vectors


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
