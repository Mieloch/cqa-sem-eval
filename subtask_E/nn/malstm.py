import pandas as pd
import numpy as np

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


def model(embeddings, max_seq_length, embedding_dim=300, n_hidden=50, gradient_clipping_norm=1.25, metrics=['accuracy', 'precision', 'recall']):
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
                   optimizer=optimizer, metrics=['accuracy', 'precision', 'recall'])

    return malstm


def train_malstm(train_data_path, test_data_path, w2v_model_path, epochs=5, batch_size=64):
    embedding_dim = 300

    print('Loading CSV data...', end=' ')
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print('Done.')

    print('Loading word2vec model...', end=' ')
    w2v_model = KeyedVectors.load_word2vec_format(w2v_model_path, binary=True)
    print('Done.')

    # Prepare vocab and embeddings matrix
    vocabulary = build_vocabulary([train_df, test_df], w2v_model)
    embeddings = build_embeddings(vocabulary, w2v_model, embedding_dim)

    # Remove word2vec model, as we don't need it anymore
    del w2v_model

    # Convert questions to number representations
    convert_questions([train_df, test_df], vocabulary)

    # Find max sequence length
    max_seq_length = get_max_seq_length([train_df, test_df])

    (X_train, Y_train), (X_validation, Y_validation) = prepare_dataset(
        train_df, max_seq_length=max_seq_length, validation_size=100)

    # Build model
    malstm = model(embeddings, max_seq_length, n_hidden=50,
                   embedding_dim=embedding_dim)

    # Setup callbacks
    checkpoint_name = 'model-{epoch: 02d}-{val_loss:.2f}.hdf5'
    if checkpoint_prefix is not None:
        checkpoint_name = checkpoint_prefix + checkpoint_name

    checkpoint = ModelCheckpoint(filepath='../models/{}'.format(checkpoint_name),
                                 period=1,
                                 save_best_only=True)

    # Training
    train_input = [X_train['left'], X_train['right']]
    validation_input = [X_validation['left'], X_validation['right']]

    trained = malstm.fit(train_input, Y_train, batch_size=batch_size, epochs=epochs,
                         validation_data=(validation_input, Y_validation),
                         callbacks=[checkpoint])


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--train-data', dest='train_data_path',
                        default='/Volumes/DataDrive/stripped/english-train-xsmall.csv')
    parser.add_argument('--test-data', dest='test_data_path',
                        default='/Volumes/DataDrive/stripped/english-devel-xsmall.csv')
    parser.add_argument('--w2v-model', dest='w2v_model_path',
                        default='/Volumes/DataDrive/models/GoogleNews-vectors-negative300.bin')

    args = parser.parse_args()

    train_malstm(args.train_data_path,
                 args.test_data_path, args.w2v_model_path)
