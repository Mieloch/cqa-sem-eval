import itertools
import re

import keras
import keras.backend as K
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from stop_words import get_stop_words
import sklearn
import argparse
from keras.models import load_model

parser = argparse.ArgumentParser()
parser.add_argument("c", help="continue training")
args = parser.parse_args()

# global variables
LSTM_N = 128
EPOCHS = 20
BATCH_SIZE = 128
TRAIN_DATA_SET_FILE_NAME = "augmented_train_set.csv"
VALIDATION_DATA_SET_FILE_NAME = "validation_data_set.csv"
EMBEDDING_FILE = "GoogleNews-vectors-negative300.bin"
RELATED_QUESTION_ID = "related_question_id"
RELATED_COMMENT_ID = "related_comment_id"
QUESTION_TEXT = "related_question_text"
COMMENT_TEXT = "related_comment_text"
RELEVANCE = "relevance"


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


def label_to_class(label):
    if label == "Good":
        return 1
    elif label == "PotentiallyUseful":
        return 0
    elif label == "Bad":
        return 0


training_data_frame = pd.read_csv(TRAIN_DATA_SET_FILE_NAME)
validation_data_frame = pd.read_csv(VALIDATION_DATA_SET_FILE_NAME)


word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
stop_words = get_stop_words('en')
questions_cols = [QUESTION_TEXT, COMMENT_TEXT]

vocabulary = {}
inverse_vocabulary = ['<unk>']  # '<unk>' will never be used, it is only a placeholder for the [0, 0, ....0] embedding

data_frames = [training_data_frame, validation_data_frame]
for data_frame in data_frames:
    for index, row in data_frame.iterrows():
        for question_col in questions_cols:
            words_to_numbers = []
            for word in text_to_words(row[question_col]):
                if word in stop_words and word not in word2vec.vocab:
                    continue
                if word not in vocabulary:
                    vocabulary[word] = len(inverse_vocabulary)
                    inverse_vocabulary.append(word)
                words_to_numbers.append(vocabulary[word])
            data_frame.set_value(index, question_col, words_to_numbers)
        data_frame.set_value(index, RELEVANCE, label_to_class(row[RELEVANCE]))

embedding_dim = 300
embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)
embeddings[0] = 0

print(training_data_frame[RELEVANCE].describe())
print("##")
print(validation_data_frame[RELEVANCE].describe())

for word, index in vocabulary.items():
    if word in word2vec.vocab:
        embeddings[index] = word2vec.word_vec(word)
del word2vec

# prepare training and test data
max_seq_length = max(training_data_frame[QUESTION_TEXT].map(lambda x: len(x)).max(),
                     training_data_frame[COMMENT_TEXT].map(lambda x: len(x)).max(),
                     validation_data_frame[QUESTION_TEXT].map(lambda x: len(x)).max(),
                     validation_data_frame[COMMENT_TEXT].map(lambda x: len(x)).max())
X_train = training_data_frame[questions_cols]
Y_train = training_data_frame[RELEVANCE]

X_validation = validation_data_frame[questions_cols]
Y_validation = validation_data_frame[RELEVANCE]

# Split to dicts
X_train = {'left': X_train[QUESTION_TEXT], 'right': X_train[COMMENT_TEXT]}
X_validation = {'left': X_validation[QUESTION_TEXT], 'right': X_validation[COMMENT_TEXT]}

# Zero padding
for dataset, side in itertools.product([X_train, X_validation], ['left', 'right']):
    dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)

# Make sure everything is ok
assert X_train['left'].shape == X_train['right'].shape
assert X_validation['left'].shape == X_validation['right'].shape
assert len(X_train['left']) == len(Y_train)
assert len(X_validation['left']) == len(Y_validation)


# define nn model
model = None
if args.c:
    model = load_model('rnn.m')
else:
    # input
    left_input = keras.layers.Input(shape=(max_seq_length,), dtype='int32')
    right_input = keras.layers.Input(shape=(max_seq_length,), dtype='int32')
    # embedding
    embedding_layer = keras.layers.Embedding(len(embeddings), embedding_dim, weights=[embeddings],
                                             input_length=max_seq_length, trainable=False)
    embedding_left = embedding_layer(left_input)
    embedding_right = embedding_layer(right_input)
    # lstm
    shared_lstm = keras.layers.LSTM(LSTM_N)
    encoded_left = shared_lstm(embedding_left)
    encoded_right = shared_lstm(embedding_right)
    # output
    merged_vector = keras.layers.concatenate([encoded_left, encoded_right], axis=-1)

    # malstm_distance = keras.layers.Merge(mode=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
    #                                      output_shape=lambda x: (x[0][0], 1))([encoded_left, encoded_right])
    x = keras.layers.Dropout(rate=0.5)(merged_vector)
    x = keras.layers.Dense(units=128,activation='elu')(x)
    x = keras.layers.Dropout(rate=0.5)(x)
    x = keras.layers.Dense(units=64,activation='elu')(x)
    x = keras.layers.Dropout(rate=0.5)(x)
    x = keras.layers.Dense(1)(x)
    model = keras.models.Model(inputs=[left_input, right_input], outputs=x)
    model.compile(optimizer=keras.optimizers.SGD(lr=0.002, momentum=0.0, decay=1e-6, nesterov=False, clipvalue=1.25),
                  loss="mean_squared_error",
                  metrics=['accuracy'])

# train
trained_model = model.fit([X_train['left'], X_train['right']], Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                          validation_data=([X_validation['left'], X_validation['right']], Y_validation))

model.save('rnn.m')
# Plot accuracy
plt.plot(trained_model.history['acc'])
plt.plot(trained_model.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig("acc.png")
plt.gcf().clear()

# Plot loss
plt.plot(trained_model.history['loss'])
plt.plot(trained_model.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig("loss.png")
plt.gcf().clear()
