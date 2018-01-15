import re
import numpy as np
import pandas as pd
import itertools

from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences


QUESTION_COLS = ['original_question_text', 'related_question_text']


def get_label_value(label):
    if label == 'PerfectMatch':
        return 1
    else:
        return 0


def get_max_seq_length(train_df, test_df):
    def len_fn(x):
        return len(x)

    return max(train_df.original_question_text.map(len_fn).max(),
               train_df.related_question_text.map(len_fn).max(),
               test_df.original_question_text.map(len_fn).max(),
               test_df.related_question_text.map(len_fn).max())


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


def build_vocabulary(train_df, test_df, w2v_model):
    '''Build vocabulary best on both train and test dataset.
    It also modifies train_df and test_df in place!
    '''
    stops = set(stopwords.words('english'))

    vocabulary = dict()
    inverse_vocabulary = ['<unk>']

    for dataset in [train_df, test_df]:
        for index, row in dataset.iterrows():
            # Iterate through the text of both questions of the row
            for question in QUESTION_COLS:
                q2n = []  # q2n -> question numbers representation

                for word in text_to_word_list(row[question]):
                    # Check for unwanted words
                    if word in stops and word not in w2v_model.vocab:
                        continue

                    if word not in vocabulary:
                        vocabulary[word] = len(inverse_vocabulary)
                        q2n.append(len(inverse_vocabulary))
                        inverse_vocabulary.append(word)
                    else:
                        q2n.append(vocabulary[word])

                # Replace questions as word to question as number representation
                dataset.set_value(index, question, q2n)

    return vocabulary


def build_embeddings(vocabulary, w2v_model, embedding_dim=300):
    # This will be the embedding matrix
    embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)
    embeddings[0] = 0  # So that the padding will be ignored

    print("Building embedding matrix...")
    # Build the embedding matrix
    for word, index in vocabulary.items():
        if word in w2v_model.vocab:
            embeddings[index] = w2v_model.word_vec(word)

    return embeddings


def load_dataset(train_path, test_path, w2v_model_path, embedding_dim=300):
    '''Load and prepare train and test dataset.

    Returns: train, validation and test set
    '''
    print('Loading CSV data...', end=' ')
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print('Done.')

    print('Loading word2vec model...', end=' ')
    w2v_model = KeyedVectors.load_word2vec_format(w2v_model_path, binary=True)
    print('Done.')

    vocabulary = build_vocabulary(train_df, test_df, w2v_model)
    embeddings = build_embeddings(vocabulary, w2v_model, embedding_dim)

    # We don't need it anymore
    del w2v_model

    return train_df, test_df, vocabulary, embeddings


def prepare_dataset(train_df, test_df, max_seq_length=None, validation_size=100):
    if max_seq_length is None:
        max_seq_length = get_max_seq_length(train_df, test_df)

    train_set_size = len(train_df) - validation_size

    X = train_df[QUESTION_COLS]
    Y = train_df['relevance'].map(get_label_value)

    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X, Y, test_size=validation_size)

    # Split to dicts
    X_train = {'left': X_train.original_question_text,
               'right': X_train.related_question_text}
    X_validation = {'left': X_validation.original_question_text,
                    'right': X_validation.related_question_text}
    X_test = {'left': test_df.original_question_text,
              'right': test_df.related_question_text}

    # Convert labels to their numpy representations
    Y_train = Y_train.values
    Y_validation = Y_validation.values
    Y_test = test_df['relevance'].map(get_label_value).values

    # Zero padding
    for dataset, side in itertools.product([X_train, X_validation, X_test], ['left', 'right']):
        dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)

    # Make sure everything is ok
    assert X_train['left'].shape == X_train['right'].shape
    assert len(X_train['left']) == len(Y_train)

    return (X_train, Y_train), (X_validation, Y_validation), (X_test, Y_test)
