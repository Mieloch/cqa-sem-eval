from subtask_C.dataframe import get_dataset
from sklearn.model_selection import train_test_split
from subtask_C.custom_metrics import save_params_csv, get_acc_plot
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Merge, Dense
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.initializers import he_normal
import keras.backend as K
from keras.optimizers import Adam, SGD, Adadelta
import time
import os
import json


TRAINING_ITERATIONS = 1


# data preparation
df = get_dataset("data/SemEval2016-Task3-CQA-QL-dev.xml", "SemEval2016-Task3-CQA-QL-dev_model")
max_sequence_len = max(df.question.map(lambda x: len(x)).max(),
                       df.comment.map(lambda x: len(x)).max())

X = df[["question", "comment"]]
y = to_categorical(df.relevance.values, num_classes=3)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train = {"question": np.array(X_train.question.tolist()), "comment": np.array(X_train.comment.tolist())}
X_test = {"question": np.array(X_test.question.tolist()), "comment": np.array(X_test.comment.tolist())}
# pad zeros
# for dataset in (X_train, X_test):
#     dataset["question"] = pad_sequences(dataset["question"], maxlen=max_sequence_len, value=np.zeros(shape=(100,)))
#     dataset["comment"] = pad_sequences(dataset["comment"], maxlen=max_sequence_len, value=np.zeros(shape=(100,)))
# test
assert X_train['question'].shape == X_train['comment'].shape
assert len(X_train['question']) == len(Y_train)


# create model
n_hidden = 256
batch_size = 1
n_epoch = 1
gradient_clipping_norm = 1.25
learning_rate = 0.001
optimizer = Adam(clipnorm=gradient_clipping_norm)
loss_f = "categorical_crossentropy"
kernel_init = he_normal()

def exponent_neg_manhattan_distance(left, right):
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

question_input = Input(shape=(None, 100))
comment_input = Input(shape=(None, 100))

shared_lstm = LSTM(n_hidden, kernel_initializer=kernel_init)

question_output = shared_lstm(question_input)
comment_output = shared_lstm(comment_input)

malstm_distance = Merge(mode=lambda x: exponent_neg_manhattan_distance(x[0], x[1]), output_shape=lambda x: (x[0][0], 1))([question_output, comment_output])

output_category = Dense(3, activation="softmax")(malstm_distance)

malstm = Model([question_input, comment_input], [output_category])




malstm.compile(loss=loss_f, optimizer=optimizer, metrics=["accuracy"])


# train and evaluate model
train_log = dict([("acc", []), ("loss", [])])
test_log = dict([("acc", []), ("loss", [])])

for i in range(1, TRAINING_ITERATIONS, 1):
    print("+++ ITERATION {}/{} +++".format(i, TRAINING_ITERATIONS))
    n_train_samples = len(X_train["question"])
    n_test_samples = len(X_test["question"])
    train_score = [0, 0]
    test_score = [0, 0]

    for j in range(n_train_samples):
        input_1 = np.array([X_train['question'][j]])
        input_2 = np.array([X_train['comment'][j]])
        y = np.array([Y_train[j]])
        malstm.fit([input_1, input_2], y, batch_size=batch_size, epochs=n_epoch, verbose=False)

        loss, acc = malstm.evaluate([input_1, input_2], y, batch_size=batch_size, verbose=False)
        train_score[0] += loss
        train_score[1] += acc

    train_score = list(map(lambda x: x / n_train_samples, train_score))
    train_log['acc'].append(train_score[1])
    train_log["loss"].append(train_score[0])

    for j in range(n_test_samples):
        input_1 = np.array([X_test['question'][j]])
        input_2 = np.array([X_test['comment'][j]])
        y = np.array([Y_test[j]])
        loss, acc = malstm.evaluate([input_1, input_2], y, batch_size=batch_size, verbose=False)
        test_score[0] += loss
        test_score[1] += acc
    test_score = list(map(lambda x: x / n_test_samples, test_score))
    test_log["acc"].append(test_score[1])
    test_log["loss"].append(test_score[0])

    # malstm.fit([X_train["question"], X_train["comment"]], Y_train, batch_size=batch_size, epochs=n_epoch)
    # print("+++ Evaluate train set +++")
    # score = malstm.evaluate([X_train["question"], X_train["comment"]], Y_train, batch_size=64)
    # train_log['acc'].append(score[1])
    # train_log["loss"].append(score[0])
    # print("+++ Evaluate test set +++")
    # score = malstm.evaluate([X_test["question"], X_test["comment"]], Y_test, batch_size=64)
    # test_log['acc'].append(score[1])
    # test_log["loss"].append(score[0])

get_acc_plot(plt, train_log, 211)
plt.title("MaLSTM - train")
get_acc_plot(plt, test_log, 212)
plt.title("MaLSTM - test")

timestamp = time.strftime("%Y%m%d_%H%M%S")
modelname = "MaLSTM"
path = "subtask_C\\models\\" + modelname + "_" + timestamp
os.makedirs(path)
model_filename = path + "\model.h5"
csvname = path + "\params.csv"
plotname = path + "\plot.png"

model_config_filename = path + "\model_config.json"
model_weights_filename = path + "\model_weights.h5"
plt.savefig(plotname, bbox_inches="tight")
params = {"iterations": str(TRAINING_ITERATIONS),
          "epochs": str(n_epoch),
          "batch size": str(batch_size),
          "n_hidden": str(n_hidden),
          "loss": loss_f,
          "optimizer": "Adam",
          "gradient clipping norm": str(gradient_clipping_norm),
          "kernel initializer": "he_normal"}
save_params_csv(csvname, params)
