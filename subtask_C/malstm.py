from subtask_C.dataframe import get_dataset
from sklearn.model_selection import train_test_split
from subtask_C.custom_metrics import precision, recall
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Merge, Dense
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K
from keras.optimizers import Adadelta
import time

TRAINING_ITERATIONS = 10


# load dataframe
df = get_dataset("data/SemEval2016-Task3-CQA-QL-dev.xml", "SemEval2016-Task3-CQA-QL-dev_model")

# get max lengths
max_sequence_len = max(df.question.map(lambda x: len(x)).max(),
                       df.comment.map(lambda x: len(x)).max())


# split X y
X = df[["question", "comment"]]
y = np.array(df.relevance.tolist())

# split into test and train sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# to dictionary
X_train = {"question": X_train.question, "comment": X_train.comment}
X_test = {"question": X_test.question, "comment": X_test.comment}

# pad zeros
for dataset in (X_train, X_test):
    dataset["question"] = pad_sequences(dataset["question"], maxlen=max_sequence_len, value=np.zeros(shape=(100,)))
    dataset["comment"] = pad_sequences(dataset["comment"], maxlen=max_sequence_len, value=np.zeros(shape=(100,)))

# Make sure everything is ok
assert X_train['question'].shape == X_train['comment'].shape
assert len(X_train['question']) == len(Y_train)

# model variables
n_hidden = 50
gradient_clipping_norm = 1.25
batch_size = 32
n_epoch = 5

def exponent_neg_manhattan_distance(left, right):
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

# The visible layer
question_input = Input(shape=(max_sequence_len, 100))
comment_input = Input(shape=(max_sequence_len, 100))

# Since this is a siamese network, both sides share the same LSTM
shared_lstm = LSTM(n_hidden)

# output layers
question_output = shared_lstm(question_input)
comment_output = shared_lstm(comment_input)

# Calculates the distance as defined by the MaLSTM model
malstm_distance = Merge(mode=lambda x: exponent_neg_manhattan_distance(x[0], x[1]), output_shape=lambda x: (x[0][0], 1))([question_output, comment_output])

# Softmax output for classification
output_category = Dense(3, activation="softmax")(malstm_distance)

# Pack it all up into a model
malstm = Model([question_input, comment_input], [output_category])
optimizer = Adadelta(clipnorm=gradient_clipping_norm)
malstm.compile(loss='mean_squared_error', optimizer=optimizer, metrics=["accuracy"])

# train model
# train_log = dict([("prec", []), ("rec", []), ("loss", [])])
# test_log = dict([("prec", []), ("rec", []), ("loss", [])])

train_log = dict([("acc", []), ("loss", [])])
test_log = dict([("acc", []), ("loss", [])])

for i in range(0, TRAINING_ITERATIONS, 1):
    print("+++ ITERATION {}/{} +++".format(i, TRAINING_ITERATIONS))
    malstm.fit([X_train['question'], X_train['comment']], Y_train, batch_size=batch_size, epochs=n_epoch)

    print("+++ Evaluate train set +++")
    score = malstm.evaluate([X_train["question"], X_train["comment"]], Y_train, batch_size=64)
    # train_log["prec"].append(score[1])
    # train_log["rec"].append(score[2])
    train_log['acc'].append(score[1])
    train_log["loss"].append(score[0])
    print("+++ Evaluate test set +++")
    score = malstm.evaluate([X_test["question"], X_test["comment"]], Y_test, batch_size=64)
    # test_log["prec"].append(score[1])
    # test_log["rec"].append(score[2])
    test_log["acc"].append(score[1])
    test_log["loss"].append(score[0])

# plot learning process
x = np.arange(0, TRAINING_ITERATIONS, step=1)

plt.subplot(211)
# train_loss, train_prec, train_rec = plt.plot(x, train_log["loss"], 'r', x, train_log["prec"], 'g', x, train_log["rec"], 'b')
# train_loss.set_label("loss (train)")
# train_prec.set_label("precision (train)")
# train_rec.set_label("recall (train)")
train_loss, train_acc = plt.plot(x, train_log["loss"], 'r', x, train_log["acc"], 'b')
train_loss.set_label("loss (train)")
train_acc.set_label("acc (train")

# plt.legend(handles=[train_loss, train_prec, train_rec], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.legend(handles=[train_loss, train_acc], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title("MaLSTM - train")

plt.subplot(212)
# test_loss, test_prec, test_rec = plt.plot(x, test_log["loss"], 'r', x, test_log["prec"], 'g', x, test_log["rec"], 'b')
# test_loss.set_label("loss (test)")
# test_prec.set_label("precision (test)")
# test_rec.set_label("recall (test)")
test_loss, test_acc = plt.plot(x, test_log["loss"], 'r', x, test_log["acc"], 'b')
test_loss.set_label("loss (test)")
test_acc.set_label("acc (test")

#plt.legend(handles=[test_loss, test_prec, test_rec], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.legend(handles=[test_loss, test_acc], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.title("MaLSTM - test")

plt.savefig("subtask_C\plots\MaLSTM_" + time.strftime("%Y%m%d_%H%M%S") + ".png", bbox_inches="tight")
