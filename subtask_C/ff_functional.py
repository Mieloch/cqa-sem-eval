import time
import os
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import itertools
import datetime

from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, Dense, concatenate, Dropout, Flatten
import keras.backend as K
from keras.optimizers import SGD


def run_training(n_epoch):
    # File paths
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    modelname = "Feedforward_1_dense"
    TRAIN_CSV = 'subtask_C\\csv_data\\train.csv'
    TEST_CSV = 'subtask_C\\csv_data\\test.csv'
    EMBEDDING_FILE = 'word2vec_model\\GoogleNews-vectors-negative300.bin.gz'
    MODEL_SAVING_DIR = "subtask_C\\models\\" + modelname + "_" + timestamp

    # Create dir
    os.makedirs(MODEL_SAVING_DIR)

    # Load training and test set
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    stops = set(stopwords.words('english'))


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


    # Prepare embedding
    vocabulary = dict()
    inverse_vocabulary = ['<unk>']  # '<unk>' will never be used, it is only a placeholder for the [0, 0, ....0] embedding
    word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

    cols = ['orgq', 'relc']

    # Iterate over the questions only of both training and test datasets
    for dataset in [train_df, test_df]:
        for index, row in dataset.iterrows():

            # Iterate through the text of both questions of the row
            for col in cols:

                q2n = []  # q2n -> question numbers representation
                for word in text_to_word_list(row[col]):

                    # Check for unwanted words
                    if word in stops and word not in word2vec.vocab:
                        continue

                    if word not in vocabulary:
                        vocabulary[word] = len(inverse_vocabulary)
                        q2n.append(len(inverse_vocabulary))
                        inverse_vocabulary.append(word)
                    else:
                        q2n.append(vocabulary[word])

                # Replace questions as word to question as number representation
                dataset.set_value(index, col, q2n)

    embedding_dim = 300
    embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)  # This will be the embedding matrix
    embeddings[0] = 0  # So that the padding will be ignored

    # Build the embedding matrix
    for word, index in vocabulary.items():
        if word in word2vec.vocab:
            embeddings[index] = word2vec.word_vec(word)

    del word2vec


    max_seq_length = max(train_df.orgq.map(lambda x: len(x)).max(),
                         train_df.relc.map(lambda x: len(x)).max(),
                         test_df.orgq.map(lambda x: len(x)).max(),
                         test_df.relc.map(lambda x: len(x)).max())

    # Split to train validation
    validation_size = 4000
    training_size = len(train_df) - validation_size

    X = train_df[cols]
    Y = train_df['relc_orgq_relevance']

    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)

    # Split to dicts
    X_train = {'left': X_train.orgq, 'right': X_train.relc}
    X_validation = {'left': X_validation.orgq, 'right': X_validation.relc}
    X_test = {'left': test_df.orgq, 'right': test_df.relc}

    # Zero padding
    for dataset, side in itertools.product([X_train, X_validation], ['left', 'right']):
        dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)

    # Make sure everything is ok
    assert X_train['left'].shape == X_train['right'].shape
    assert len(X_train['left']) == len(Y_train)


    # Model variables
    n_hidden = 128
    batch_size = 64

    # The visible layer
    left_input = Input(shape=(max_seq_length,), dtype='int32')
    right_input = Input(shape=(max_seq_length,), dtype='int32')

    embedding_layer = Embedding(len(embeddings), embedding_dim, weights=[embeddings],
                                input_length=max_seq_length, trainable=False)

    # Embedded version of the inputs
    encoded_left = embedding_layer(left_input)
    encoded_right = embedding_layer(right_input)

    concatenated = concatenate([encoded_left, encoded_right], axis=-1)

    dense_1 = Dense(n_hidden, activation="elu", kernel_initializer="he_normal")(concatenated)
    drop_1 = Dropout(0.5)(dense_1)
    #dense_2 = Dense(n_hidden, activation="elu", kernel_initializer="he_normal")(drop_1)
    #drop_2 = Dropout(0.5)(dense_2)
    #dense_3 = Dense(64, activation="elu", kernel_initializer="he_normal")(drop_2)
    #drop_3 = Dropout(0.5)(dense_3)
    flat = Flatten()(drop_1)
    output = Dense(1)(flat)

    feedforward = Model([left_input, right_input], output)

    optimizer = SGD(lr=0.005, decay=1e-6, clipvalue=1.25)

    feedforward.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

    # Start training
    training_start_time = time.time()

    lstm_trained = feedforward.fit([X_train['left'], X_train['right']], Y_train, batch_size=batch_size, epochs=n_epoch,
                                   validation_data=([X_validation['left'], X_validation['right']], Y_validation))

    print("Training time finished.\n{} epochs in {}".format(n_epoch, datetime.timedelta(
        seconds=time.time() - training_start_time)))

    # Plot accuracy
    plt.plot(lstm_trained.history['acc'])
    plt.plot(lstm_trained.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    # plt.show()
    plt.savefig(MODEL_SAVING_DIR + "\\acc_plot.png")
    plt.close()

    # Plot loss
    plt.plot(lstm_trained.history['loss'])
    plt.plot(lstm_trained.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    # plt.show()
    plt.savefig(MODEL_SAVING_DIR + "\\loss_plot.png")
    plt.close()

run_training(400)
