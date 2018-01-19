import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors

from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Merge, Lambda
import keras.backend as K
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint

from nn_utils import text_to_word_list, get_max_seq_length, prepare_dataset
from nn_utils import build_embeddings, build_vocabulary, convert_questions


DATA_DIR = '/Volumes/DataDrive'
EMBEDDING_FILE = DATA_DIR + '/models/GoogleNews-vectors-negative300.bin'
TRAIN_CSV = DATA_DIR + '/stripped/english-train-xsmall.csv'
TEST_CSV = DATA_DIR + '/stripped/english-devel-xsmall.csv'


def exponent_neg_manhattan_distance(lay):
    '''Overriden exponent manhattan distance function'''
    return K.exp(-K.sum(K.abs(lay[0] - lay[1]), axis=1, keepdims=True))


def my_out_shape(shapes):
    return (shapes[0][0], 1)


def f2_score(y_true, y_pred):
    y_true = tf.cast(y_true, "int32")
    # implicit 0.5 threshold via tf.round
    y_pred = tf.cast(tf.round(y_pred), "int32")
    y_correct = y_true * y_pred
    sum_true = tf.reduce_sum(y_true, axis=1)
    sum_pred = tf.reduce_sum(y_pred, axis=1)
    sum_correct = tf.reduce_sum(y_correct, axis=1)
    precision = sum_correct / sum_pred
    recall = sum_correct / sum_true
    f_score = 5 * precision * recall / (4 * precision + recall)
    f_score = tf.where(tf.is_nan(f_score), tf.zeros_like(f_score), f_score)
    return tf.reduce_mean(f_score)


def f1_score(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0 or c2 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def model(embeddings, max_seq_length, embedding_dim=300, n_hidden=50, gradient_clipping_norm=1.25, metrics=['accuracy']):
    '''Build MaLSTM network model'''

    # The visible layer
    left_input = Input(shape=(max_seq_length,), dtype='int32')
    right_input = Input(shape=(max_seq_length,), dtype='int32')

    embedding_layer = Embedding(
        len(embeddings), embedding_dim, weights=[embeddings],
        input_length=max_seq_length, trainable=False)

    # Embedded version of the inputs
    encoded_left = embedding_layer(left_input)
    encoded_right = embedding_layer(right_input)

    # Since this is a siamese network, both sides share the same LSTM
    shared_lstm = LSTM(n_hidden)

    left_output = shared_lstm(encoded_left)
    right_output = shared_lstm(encoded_right)

    # Calculates the distance as defined by the MaLSTM model
    malstm_distance = Lambda(exponent_neg_manhattan_distance,
                             output_shape=my_out_shape)([left_output, right_output])

    # Pack it all up into a model
    malstm = Model([left_input, right_input], [malstm_distance])

    # Adadelta optimizer, with gradient clipping by norm
    optimizer = Adadelta(clipnorm=gradient_clipping_norm)

    malstm.compile(loss='mean_squared_error',
                   optimizer=optimizer, metrics=metrics)

    return malstm
