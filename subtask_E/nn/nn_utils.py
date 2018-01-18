import re
import numpy as np
import pandas as pd
import itertools
import pickle
import os
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences


QUESTION_COLS = ['original_question_text', 'related_question_text']


def get_label_value(label):
    if label == 'PerfectMatch':
        return 1
    else:
        return 0


def get_max_seq_length(dataframes):
    def len_fn(x):
        return len(x)

    max_len = 0
    for df in dataframes:
        max_df_len = max(df.original_question_text.map(len_fn).max(),
                         df.related_question_text.map(len_fn).max())
        max_len = max(max_df_len, max_len)

    return max_len


def text_to_word_list(text):
    ''' Pre process and convert texts to a list of words '''
    text = str(text)
    text = text.lower()

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
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.split()

    return text


def load_vocabulary(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            'Couldn\'t find vocabulary file: {}'.format(filepath))

    with open(filepath, 'rb') as fp:
        vocabulary = pickle.load(fp)
        return vocabulary


def build_vocabulary(dataframes, w2v_model, save_to=None):
    '''Build vocabulary best on both train and test dataset.
    It also modifies train_df and test_df in place!
    '''
    stops = set(stopwords.words('english'))

    vocabulary = dict()
    inverse_vocabulary = ['<unk>']

    for dataset in dataframes:
        for index, row in dataset.iterrows():
            # Iterate through the text of both questions of the row
            for question in QUESTION_COLS:
                for word in text_to_word_list(row[question]):
                    # Check for unwanted words
                    if word in stops and word not in w2v_model.vocab:
                        continue

                    if word not in vocabulary:
                        vocabulary[word] = len(inverse_vocabulary)
                        inverse_vocabulary.append(word)

    if save_to is not None:
        if os.path.exists(save_to):
            raise FileExistsError('Cannot overwrite {} file'.format(save_to))

        with open(save_to, 'wb') as fp:
            pickle.dump(vocabulary, fp)

    return vocabulary


def convert_questions(dataframes, vocabulary):
    '''Converts original and related questions into number representation based on vocabulary'''
    for dataset in dataframes:
        for index, row in dataset.iterrows():
            for question in QUESTION_COLS:
                q2n = []

                for word in text_to_word_list(row[question]):
                    if word not in vocabulary:
                        continue

                    q2n.append(vocabulary[word])

                dataset.at[index, question] = q2n


def load_embeddings(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            'Couldn\'t find {} embeddings'.format(filepath))

    return np.load(filepath)


def build_embeddings(vocabulary, w2v_model, embedding_dim=300, save_to=None):
    # This will be the embedding matrix
    embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)
    embeddings[0] = 0  # So that the padding will be ignored

    print("Building embedding matrix...")
    # Build the embedding matrix
    for word, index in vocabulary.items():
        if word in w2v_model.vocab:
            embeddings[index] = w2v_model.word_vec(word)

    if save_to is not None:
        if os.path.exists(save_to):
            raise FileExistsError('Cannot overwrite {} file'.format(save_to))

        np.save(save_to, embeddings)

    return embeddings


def make_pairs(df, max_seq_length):
    left = pad_sequences(df.original_question_text.values,
                         maxlen=max_seq_length)
    right = pad_sequences(df.related_question_text.values,
                          maxlen=max_seq_length)
    return {'left': left, 'right': right}


def prepare_test_dataset(test_df, max_seq_length):
    X_test = make_pairs(test_df, max_seq_length)
    Y_test = test_df.relevance.map(get_label_value).values
    return X_test, Y_test


def prepare_dataset(train_df, max_seq_length, validation_size=100):
    train_set_size = len(train_df) - validation_size

    X = train_df[QUESTION_COLS]
    Y = train_df.relevance.map(get_label_value)

    # Split dataset into train and validation set
    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X, Y, test_size=validation_size)

    # Split to dicts
    X_train = make_pairs(X_train, max_seq_length)
    X_validation = make_pairs(X_validation, max_seq_length)

    # Convert labels to their numpy representations
    Y_train = Y_train.values
    Y_validation = Y_validation.values

    # Make sure everything is ok
    assert X_train['left'].shape == X_train['right'].shape
    assert len(X_train['left']) == len(Y_train)

    return (X_train, Y_train), (X_validation, Y_validation)


def plot_history(trained):
    # Plot accuracy
    plt.plot(trained.history['acc'])
    plt.plot(trained.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot loss
    plt.plot(trained.history['loss'])
    plt.plot(trained.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()
