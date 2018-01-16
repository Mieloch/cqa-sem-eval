import os
import pandas as pd
import matplotlib.pyplot as plt
from nn.nn_utils import text_to_word_list
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from functools import reduce


def row_by_label(df, label):
    return df[df.relevance == label]


def split_rows_by_label(df):
    perfect_match = row_by_label(df, 'PerfectMatch')
    related = row_by_label(df, 'Related')
    irrelevant = row_by_label(df, 'Irrelevant')

    return perfect_match, related, irrelevant


def join_sentence(sentence):
    first_element = sentence[0]
    return reduce(lambda a, b: a + ' ' + b, sentence[1:], first_element)


def generate_new_sentences(sentence, w2v_model, max_len=100, threshold=0.7):
    stops = set(stopwords.words('english'))
    word_list = text_to_word_list(sentence)
    new_sentences_count = 0

    for index, word in enumerate(word_list):
        if new_sentences_count >= max_len:
            break

        if word in stops or word not in w2v_model.vocab:
            continue

        similar_word, score = w2v_model.similar_by_word(word, topn=1)[0]
        if score >= threshold:
            new_sentence = word_list[:]
            new_sentence[index] = similar_word
            new_sentence = join_sentence(new_sentence)

            yield new_sentence
            new_sentences_count += 1


def extend_dataframe(df, w2v_model, verbose=False):
    for index, row in df.iterrows():
        sentence = row.original_question_text

        for new_sentence in generate_new_sentences(sentence, w2v_model):
            new_row = row.copy()
            new_row.original_question_id = 'n/a'  # it's a new question.
            new_row.original_question_text = new_sentence
            df = df.append(new_row, ignore_index=True)

        if verbose:
            print('New PerfectMatch size = {}'.format(len(df)))

    return df


def build_extended_dataframe(df, w2v_model, shuffle=False, save_to=None, verbose=False):
    perfect_match, related, irrelevant = split_rows_by_label(df)

    # Extend "PerfectMatch" set
    extended_perfect_match = extend_dataframe(
        perfect_match.copy(), w2v_model, verbose=verbose)

    # Build new dataframe
    new_df = extend_perfect_match.append(related, ignore_index=True)
    new_df = new_df.append(irrelevant[:len(extended_perfect_match) * 3])

    if shuffle:
        new_df = new_df.sample(frac=1)

    # Save dataset to csv file (prevent overwriting)
    if save_to is not None:
        if os.path.exists(save_to):
            raise FileExistsError('Save to operation would overwrite existing {} file. Aborting.'.format(save_to))

        new_df.to_csv(save_to, index=False)

    return new_df
