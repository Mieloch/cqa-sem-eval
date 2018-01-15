from time import time
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# import seaborn as sns

import itertools
import datetime

from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Merge, Lambda
import keras.backend as K
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint
from nn_utils import text_to_word_list


EMBEDDING_FILE = '/Volumes/DataDrive/models/GoogleNews-vectors-negative300.bin'
TRAIN_CSV = '/Volumes/DataDrive/stripped/english-train-xsmall.csv'
TEST_CSV = '/Volumes/DataDrive/stripped/english-devel-xsmall.csv'


def model():
