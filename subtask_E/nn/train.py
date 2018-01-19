import argparse
import pandas as pd
import numpy as np

from gensim.models import KeyedVectors
from keras.callbacks import CSVLogger, ModelCheckpoint

from nn_utils import build_embeddings, build_vocabulary, convert_questions, get_max_seq_length, prepare_dataset
from malstm import model, f2_score


def train_malstm(data_dir, epochs=5, batch_size=64, validation_size=3000, session_name='malstm-training'):
    print('Starting "{}" training session...'.format(session_name))

    embeddings_dim = 300

    print('Loading data...', sep=' ', end='', flush=True)
    train_df = pd.read_csv(data_dir + '/merged/en-train-extended.csv')
    test_df = pd.read_csv(data_dir + '/merged/en-test.csv')
    print('Done.')

    # Load word2vec model trained on GoogleNews dataset
    print('Loading word2vec model...', sep=' ', end='', flush=True)
    w2v_model = KeyedVectors.load_word2vec_format(
        data_dir + '/models/GoogleNews-vectors-negative300.bin', binary=True)
    print('Done.')

    # Prepare vocab and embeddings matrix
    print('Building vocabulary...', sep=' ', end='', flush=True)
    vocabulary = build_vocabulary([train_df, test_df], w2v_model)
    print('Done.')

    print('Building embeddings...', sep=' ', end='', flush=True)
    embeddings = build_embeddings(vocabulary, w2v_model, embeddings_dim)
    print('Done.')

    # Remove word2vec model, as we don't need it anymore
    del w2v_model

    # Convert questions to number representations
    print('Converting questions...', sep=' ', end='', flush=True)
    convert_questions([train_df, test_df], vocabulary)
    print('Done.')

    # Find max sequence length
    max_seq_length = get_max_seq_length([train_df, test_df])

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
        data_dir + '/training/logs/{}.csv'.format(session_name))

    checkpoint_path = data_dir + \
        '/training/models/' + session_name + \
        '-{epoch: 02d}-{val_loss:.2f}.hdf5'
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
