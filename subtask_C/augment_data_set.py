#!/usr/bin/env python
import copy
import re

import nltk
import pandas as pd
from gensim.models import KeyedVectors
from stop_words import get_stop_words

FILE_NAME = "subtask_C\\csv_data\\train.csv"
EMBEDDING_FILE = "word2vec_model\\GoogleNews-vectors-negative300.bin.gz"
QUESTION_TEXT = "orgq"
COMMENT_TEXT = "relc"
RELEVANCE = "relc_orgq_relevance"


# create embedding matrix

def text_to_words(text):
    text = str(text).lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    words = nltk.word_tokenize(text)
    return words


questions_cols = [QUESTION_TEXT, COMMENT_TEXT]
question_cols = [COMMENT_TEXT]
word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
stop_words = get_stop_words('en')
data_frame = pd.read_csv(FILE_NAME)

print("original data_frame size : {}".format(len(data_frame)))
cp = data_frame.copy()

for index, row in cp.iterrows():
    if row[RELEVANCE] == 1:
        for question_col in questions_cols:
            similarities = {}
            v = []
            for word in text_to_words(row[question_col]):
                if word not in stop_words and word in word2vec.vocab:
                    synonym, similarity = word2vec.similar_by_word(word, topn=1)[0]
                    if similarity > 0.7:
                        similarities[word] = {'word': word, 'synonym': synonym, 'similarity': similarity}
            similarities_list = list(similarities.values())
            similarities_list.sort(key=lambda x: x["similarity"], reverse=True)
            for best in similarities_list[0:3]:
                with_similar = str(row[question_col]).lower().replace(best['word'], best['synonym'])
                new_row = copy.deepcopy(row)
                new_row[question_col] = with_similar
                data_frame = data_frame.append(new_row)
        if index % 100 == 0:
            print("augmented data_frame size : {}".format(len(data_frame)))
            data_frame.to_csv("subtask_C\\csv_data\\good_augmented_train.csv", encoding="utf-8")
print("augmented data_frame size : {}".format(len(data_frame)))
data_frame.to_csv("subtask_C\\csv_data\\good_augmented_train.csv", encoding="utf-8")
