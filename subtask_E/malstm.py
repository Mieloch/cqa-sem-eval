import numpy as np
import keras.backend as K
import pandas as pd
from keras.models import Model
from keras.layers import Input, LSTM, Merge, Dense
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam

def load_dataset(train_csv_path, test_csv_path):
    train_df = pd.read_csv(train_csv_path)
    X_train = train_df[['original_question_text', 'related_question_text']]
    Y_train = train_df['relevance']

    test_df = pd.read_csv(test_csv_path)
    X_test = test_df[['original_question_text', 'related_question_text']]
    Y_test = test_df['relevance']

    return (X_train, Y_train, X_test, Y_test)


def create_model():
    X_train, Y_train, X_test, Y_test = load_dataset('/Volumes/DataDrive/stripped/english-train.csv',
                                                    '/Volumes/DataDrive/stripped/english-devel.csv')

    print(X_train.shape, Y_train.shape)



def main():
    create_model()


if __name__ == '__main__':
    main()
