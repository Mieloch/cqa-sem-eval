import argparse
import pandas as pd
import numpy as np
import os
import pathlib

from gensim.models import KeyedVectors
from keras.callbacks import CSVLogger, ModelCheckpoint

from nn_utils import \
    build_embeddings, build_vocabulary, convert_questions, \
    get_max_seq_length, prepare_dataset, load_vocabulary, \
    load_embeddings, load_max_seq_length
from malstm import model, f2_score


W2V_MODEL_PATH = 'models/GoogleNews-vectors-negative300.bin'
TRAIN_SET_PATH = 'merged/en-train-extended.csv'
TEST_SET_PATH = 'merged/en-test.csv'
TRAINING_LOGS_DIR = 'training/logs'
TRAINING_MODELS_DIR = 'training/models'
MISC_DIR = 'training/misc'
VOCABULARY_PATH = MISC_DIR + '/vocabulary.pickle'
EMBEDDINGS_PATH = MISC_DIR + '/embeddings.pickle'
MAX_SEQ_LENGTH_PATH = MISC_DIR + '/max_seq_length.pickle'


def check_env(data_dir):
    model_file_exists = os.path.exists(
        os.path.join(data_dir, W2V_MODEL_PATH))
    train_file_exists = os.path.exists(
        os.path.join(data_dir, TRAIN_SET_PATH))
    test_file_exists = os.path.exists(
        os.path.join(data_dir, TEST_SET_PATH))

    if not model_file_exists:
        raise FileNotFoundError('Model file was not found.')

    if not train_file_exists:
        raise FileNotFoundError('Training file was not found.')

    if not test_file_exists:
        raise FileNotFoundError('Test file was not found.')

    pathlib.Path(os.path.join(
        data_dir, TRAINING_LOGS_DIR)).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(
        data_dir, TRAINING_MODELS_DIR)).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(
        data_dir, MISC_DIR)).mkdir(parents=True, exist_ok=True)


def train_malstm(data_dir, epochs=5, batch_size=64, validation_size=3000, session_name='malstm-training'):
    # Check environment for data files and make directories if they not
    check_env(data_dir)

    print('Starting "{}" training session...'.format(session_name))

    embeddings_dim = 300

    print('Loading data...', sep=' ', end='', flush=True)
    train_df = pd.read_csv(os.path.join(data_dir, TRAIN_SET_PATH))
    test_df = pd.read_csv(os.path.join(data_dir, TEST_SET_PATH))
    print('Done.')

    # Prepare vocab and embeddings matrix
    vocabulary_full_path = os.path.join(data_dir, VOCABULARY_PATH)
    embeddings_full_path = os.path.join(data_dir, EMBEDDINGS_PATH)

    # Load word2vec model trained on GoogleNews dataset
    should_load_w2v_model = (not os.path.exists(vocabulary_full_path)) or (
        not os.path.exists(embeddings_full_path))

    if should_load_w2v_model:
        print('Loading word2vec model...', sep=' ', end='', flush=True)
        w2v_model = KeyedVectors.load_word2vec_format(
            os.path.join(data_dir, W2V_MODEL_PATH), binary=True)
        print('Done.')

    try:
        vocabulary = load_vocabulary(vocabulary_full_path)
    except FileNotFoundError:
        print('Vocabulary file not found. Building vocabulary...',
              sep=' ', end='', flush=True)
        vocabulary = build_vocabulary(
            [train_df, test_df], w2v_model, save_to=vocabulary_full_path)
        print('Done.')

    try:
        embeddings = load_embeddings(embeddings_full_path)
    except FileNotFoundError:
        print('Embeddings file not found. Building embeddings...', sep=' ', end='', flush=True)
        embeddings = build_embeddings(vocabulary, w2v_model, embeddings_dim, save_to=embeddings_full_path)
        print('Done.')

    # Remove word2vec model, as we don't need it anymore
    if should_load_w2v_model:
        del w2v_model

    # Convert questions to number representations
    print('Converting questions...', sep=' ', end='', flush=True)
    convert_questions([train_df, test_df], vocabulary)
    print('Done.')

    # Find max sequence length
    max_seq_length_full_path = os.path.join(data_dir, MAX_SEQ_LENGTH_PATH)
    try:
        max_seq_length = load_max_seq_length(max_seq_length_full_path)
    except FileNotFoundError:
        max_seq_length = get_max_seq_length([train_df, test_df], save_to=max_seq_length_full_path)

    # Split dataset
    (X_train, Y_train), (X_validation, Y_validation) = prepare_dataset(
        train_df, max_seq_length=max_seq_length, validation_size=3000)

    print('X_train size={}, X_validation size={}'.format(
        len(X_train['left']), len(X_validation['left'])))

    # Build model
    print('Building model...', sep=' ', end='', flush=True)
    malstm = model(embeddings, max_seq_length, n_hidden=50,
                   embedding_dim=embeddings_dim, metrics=['accuracy', 'mae', f2_score])
    print('Done.')

    # Setup callbacks
    csv_logger = CSVLogger(
        os.path.join(data_dir, TRAINING_LOGS_DIR, '{}.csv'.format(session_name)))

    checkpoint_path = os.path.join(data_dir, TRAINING_MODELS_DIR,
        session_name + '-{epoch:02d}-{val_loss:.2f}.hdf5')
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path, period=1, save_best_only=True)

    callbacks = [csv_logger, checkpoint]

    # Training
    train_input = [X_train['left'], X_train['right']]
    validation_input = [X_validation['left'], X_validation['right']]

    trained = malstm.fit(train_input, Y_train, batch_size=batch_size, epochs=epochs,
                         validation_data=(validation_input, Y_validation),
                         callbacks=callbacks)

    print('Session "{}" has finished successfully.'.format(session_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--session-name', dest='session_name', default='malstm-training')
    parser.add_argument('--data-dir', dest='data_dir',
                        default='/Volumes/DataDrive')
    parser.add_argument('--epochs', dest='epochs', type=int, default=10)
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=64)
    parser.add_argument('--validation-size', dest='validation_size',
                        type=int, default=3000)

    args = parser.parse_args()

    train_malstm(args.data_dir,
                 epochs=args.epochs,
                 batch_size=args.batch_size,
                 validation_size=args.validation_size,
                 session_name=args.session_name)
