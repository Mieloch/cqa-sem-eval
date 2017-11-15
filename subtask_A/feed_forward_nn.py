from subtask_A.data_set import subtask_A_word2vec_dataset
import numpy as np
import keras
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import gensim
from word2vec_model.word2vec_utils import load_word2vec_model

TRAINING_ITERATIONS = 100

# load data set in word2vec representation
data_set = subtask_A_word2vec_dataset("../data/SemEval2016-Task3-CQA-QL-dev.xml",
                                      load_word2vec_model("GoogleNews-vectors-negative300.bin"))

# prepare features and labels
X = []
Y = []
for sample in data_set:
    question_vec = sample["question"]
    comment_vec = sample["comment"]
    INPUT_DIM = len(question_vec) + len(comment_vec)
    concatenated_vec = np.concatenate((question_vec, comment_vec), axis=0)
    X.append(concatenated_vec)
    Y.append(sample["relevance"])
X = np.reshape(X, (-1, INPUT_DIM))
Y = np.reshape(Y, (-1, 1))
Y = keras.utils.to_categorical(Y, num_classes=3)

# split into test and train sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# define nn model
model = keras.Sequential()
model.add(Dense(128, activation="elu", kernel_initializer=keras.initializers.he_normal(), input_dim=INPUT_DIM))
model.add(Dropout(0.5))
model.add(Dense(128, activation="elu", kernel_initializer=keras.initializers.he_normal()))
model.add(Dropout(0.5))
model.add(Dense(64, activation="elu", kernel_initializer=keras.initializers.he_normal()))
model.add(Dropout(0.5))
model.add(Dense(3, activation="softmax"))

model.compile(optimizer=keras.optimizers.SGD(lr=0.03, momentum=0.0, decay=0.0, nesterov=False),
              loss="categorical_crossentropy",
              metrics=['accuracy'])

# train model
train_log = dict([("acc", []), ("loss", [])])
test_log = dict([("acc", []), ("loss", [])])

for i in range(0, TRAINING_ITERATIONS, 1):
    print("+++ ITERATION {}/{} +++".format(i, TRAINING_ITERATIONS))
    model.fit(X_train, Y_train, epochs=5, batch_size=32)

    print("+++ Evaluate train set +++")
    score = model.evaluate(X_train, Y_train, batch_size=64)
    train_log["acc"].append(score[1])
    train_log["loss"].append(score[0])
    print("+++ Evaluate test set +++")
    score = model.evaluate(X_test, Y_test, batch_size=64)
    test_log["acc"].append(score[1])
    test_log["loss"].append(score[0])

# plot learning process
x = np.arange(0, len(test_log["acc"]), step=1)
plt.plot(x, test_log["loss"], 'r--', x, test_log["acc"], 'r', x, train_log["loss"], 'g--', x, train_log["acc"], 'g')
plt.title("Feedforward")
plt.savefig("plot.png")
