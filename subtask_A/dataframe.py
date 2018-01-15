import os
import re
from itertools import filterfalse
import gensim
import nltk
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from stop_words import get_stop_words



def tokenize_to_lower_case(sentence):
    tokenize = nltk.word_tokenize(sentence)
    return [t.lower() for t in tokenize]

def load(file_name):
    with open(file_name, 'r', encoding="utf8") as myfile:
        return BeautifulSoup(myfile.read(), "xml")

def load_word2vec_model(model_name):
    print("Loading word2vec model {}".format(model_name))
    path = "../word2vec_model/"
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


def remove_subject_from_question(question):
    return re.sub('.+\/\/\ ', '', question)


def append_zero_padding(sentence_list, max_padding_length):
    result = []
    for sentence in sentence_list:
        padding_size = max_padding_length - len(sentence)
        zeros = np.zeros((padding_size, len(sentence[0])))
        result.append(np.concatenate((sentence, zeros)))
    return result


def get_dataset(xml_file, model_name, add_zero_padding=False):
    org_qs, rel_cs, rels = get_subtask_A_data_from_file(xml_file, model_name)
    if add_zero_padding == True:
        max_question_length = (max(map(lambda x: len(x), org_qs)))
        max_comment_length = (max(map(lambda x: len(x), rel_cs)))
        max_padding_length = max(max_question_length, max_comment_length)
        org_qs = append_zero_padding(org_qs, max_padding_length)
        rel_cs = append_zero_padding(rel_cs, max_padding_length)
    questions = pd.Series(org_qs)
    comments = pd.Series(rel_cs)

    relevances = pd.Series(rels)
    dataset = pd.concat((pd.DataFrame(series) for series in (questions, comments, relevances)), axis=1)
    dataset.columns = ["question", "comment", "relevance"]
    return dataset


def get_subtask_A_data_from_file(xml_file, model_name):
    soup = load(xml_file)
    threads = soup.findAll('Thread', recursive=True)
    questions_data = []
    comments_data = []
    relevances_data = []
    word2vec_model = load_word2vec_model(model_name)
    for thread in threads:
        thread_soup = BeautifulSoup(str(thread), "xml")
        question = thread_soup.RelQuestion.RelQBody.text
        if question == '':
            print("WARN! empty question")
            continue
        comments = thread_soup.findAll("RelComment")
        for comment in comments:
            comment_text = comment.RelCText.text
            if comment_text == '':
                print("WARN! empty comment")
                continue
            question_vectors = text_to_word_vectors(remove_subject_from_question(question), word2vec_model)
            comment_vectors = text_to_word_vectors(comment_text, word2vec_model)
            if len(question_vectors) > 0 and len(comment_vectors) > 0:
                questions_data.append(question_vectors)
                comments_data.append(comment_vectors)
                relevances_data.append(label_to_class(comment['RELC_RELEVANCE2RELQ']))

    return questions_data, comments_data, relevances_data


def text_to_word_vectors(text, word2vec_model):
    vector_list = []
    tokens = tokenize_to_lower_case(text)
    tokens[:] = filterfalse(token_in_stop_words, tokens)
    for token in tokens:
        word_vector = word2vec_model.wv[token]
        vector_list.append(word_vector)
    return vector_list


stop_words = get_stop_words('en')


def token_in_stop_words(token):
    return token in stop_words


def label_to_class(label):
    if label == "Good":
        return 0
    elif label == "PotentiallyUseful":
        return 1
    elif label == "Bad":
        return 2


# data = get_dataset("..\data\dev-Q1-sample.xml", "SemEval2016-Task3-CQA-QL-dev_model",
#                    add_zero_padding=True)
# values = data.relevance.values
# print(values.shape)

#
# print(data.question.values.shape)
# print(data.question.values[0].shape)
# #