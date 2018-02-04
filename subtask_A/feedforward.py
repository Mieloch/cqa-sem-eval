import re

import keras
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from stop_words import get_stop_words

# global variables
HIDDEN = 60
EPOCHS = 30
BATCH_SIZE = 64
TRAIN_DATA_SET_FILE_NAME = "csv/train_data_set.csv"
VALIDATION_DATA_SET_FILE_NAME = "csv/validation_data_set.csv"
EMBEDDING_FILE = "../word2vec_model/GoogleNews-vectors-negative300.bin"
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


training_data_frame = pd.read_csv(TRAIN_DATA_SET_FILE_NAME)[0:100]
validation_data_frame = pd.read_csv(VALIDATION_DATA_SET_FILE_NAME)[0:100]

word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
stop_words = get_stop_words('en')
questions_cols = [QUESTION_TEXT, COMMENT_TEXT]
VECTOR_DIM = 300

data_frames = [training_data_frame, validation_data_frame]
for data_frame in data_frames:
    for index, row in data_frame.iterrows():
        for question_col in questions_cols:
            word_vectors = []
            for word in text_to_words(row[question_col]):
                if word in stop_words and word not in word2vec.vocab:
                    continue
                try:
                    vector = word2vec.word_vec(word)
                    word_vectors.append(vector)
                except Exception as E:
                    print(E)
            if len(word_vectors) == 0:
                word_vectors = np.zeros((1, 300))
            data_frame.set_value(index, question_col, np.average(word_vectors, axis=0))
        data_frame.set_value(index, RELEVANCE, label_to_class(row[RELEVANCE]))

print(training_data_frame[RELEVANCE].describe())
print("##")
print(validation_data_frame[RELEVANCE].describe())

del word2vec

# prepare training and test data
X_train = training_data_frame[questions_cols]
Y_train = training_data_frame[RELEVANCE]

X_validation = validation_data_frame[questions_cols]
Y_validation = validation_data_frame[RELEVANCE]
print("shape")
# Split to dicts
X_train = {'left': np.reshape([list(x) for x in X_train[QUESTION_TEXT].values], (-1, 300)),
           'right': np.reshape([list(x) for x in X_train[COMMENT_TEXT].values], (-1, 300))}
X_validation = {'left': np.reshape([list(x) for x in X_validation[QUESTION_TEXT].values], (-1, 300)),
                'right': np.reshape([list(x) for x in X_validation[COMMENT_TEXT].values], (-1, 300))}

print(X_train['left'].shape)

# Make sure everything is ok
assert X_train['left'].shape == X_train['right'].shape
assert X_validation['left'].shape == X_validation['right'].shape
assert len(X_train['left']) == len(Y_train)
assert len(X_validation['left']) == len(Y_validation)

# define nn model

# input
left_input = keras.layers.Input(name="question_vector", shape=[VECTOR_DIM], dtype='float32')
right_input = keras.layers.Input(name="answer_vector", shape=[VECTOR_DIM], dtype='float32')
# concatenate
merged_vector = keras.layers.concatenate(name="vectors_concatenation", inputs=[left_input, right_input], axis=-1)

# hidden
x = keras.layers.Dense(name="hidden_layer", units=HIDDEN, activation='elu')(merged_vector)
x = keras.layers.Dense(1, name="output")(x)

model = keras.models.Model(inputs=[left_input, right_input], outputs=x)
model.compile(optimizer=keras.optimizers.SGD(lr=0.0015, momentum=0.0, decay=1e-6, nesterov=False, clipvalue=1.25),
              loss="mean_squared_error",
              metrics=['accuracy'])

trained_model = model.fit([X_train['left'], X_train['right']], Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                          validation_data=([X_validation['left'], X_validation['right']], Y_validation))

from keras.utils import plot_model

plot_model(model, to_file='model.png')

predict_train = model.predict([X_train['left'], X_train['right']])
predict_train = [int(round(x)) for x in (predict_train.flatten().clip(min=0))]
Y_train = [int(x) for x in Y_train.values]
print("Train set accuracy score =", accuracy_score(Y_train, predict_train))
print("Train set recall score =", recall_score(Y_train, predict_train))
print("Train set precission score =", precision_score(Y_train, predict_train))

predict_validation = model.predict([X_validation['left'], X_validation['right']])
predict_validation = [int(round(x)) for x in predict_validation.flatten().clip(min=0)]
Y_validation = [int(x) for x in Y_validation.values]
print("Test set accuracy score =", accuracy_score(Y_validation, predict_validation))
print("Test set recall score =", recall_score(Y_validation, predict_validation))
print("Test set precission score =", precision_score(Y_validation, predict_validation))

model.save('feedforward.m')

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
