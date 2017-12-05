import numpy as np
import keras
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from subtask_A.dataframe import get_dataset

flatten = lambda l: [item for sublist in l for item in sublist]

TRAINING_ITERATIONS = 3

# load data set in word2vec representation
data_set = get_dataset("..\data\SemEval2016-Task3-CQA-QL-dev.xml", "SemEval2016-Task3-CQA-QL-dev_model",
                       add_zero_padding=True)

# prepare features and labels

X = (list(zip(data_set.question.values, data_set.comment.values)))
Y = keras.utils.to_categorical(data_set.relevance.values, num_classes = 3)

VECTOR_DIM = 100

# split into test and train sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
X_q_train, X_c_train = zip(*X_train)
X_q_test, X_c_test = zip(*X_test)

X_q_train = np.reshape(X_q_train, (-1, 204, 100))
X_c_train = np.reshape(X_c_train, (-1, 204, 100))
X_q_test = np.reshape(X_q_test, (-1, 204, 100))
X_c_test = np.reshape(X_c_test, (-1, 204, 100))
Y_train = np.reshape(Y_train, (-1, 3))
# Y_test = np.reshape(Y_test, (-1, 3))

print(np.array(X_q_train).shape)
print(np.array(Y_train).shape)
print(np.array(Y_train[0]).shape)
# define nn model
question_input = keras.Input(shape=(204, 100))
comment_input = keras.Input(shape=(204, 100))

shared_lstm = keras.layers.LSTM(units=128,
                                kernel_initializer=keras.initializers.he_normal())


encoded_question = shared_lstm(question_input)
encoded_question = keras.layers.advanced_activations.LeakyReLU()(encoded_question)
encoded_comment = shared_lstm(comment_input)
encoded_comment = keras.layers.advanced_activations.LeakyReLU()(encoded_comment)

merged_vector = keras.layers.concatenate([encoded_question, encoded_comment], axis=-1)

x = Dense(128, activation='relu')(merged_vector)

predictions = Dense(3, activation="softmax")(x)

model = keras.models.Model(inputs=[question_input, comment_input], outputs=predictions)

model.compile(optimizer=keras.optimizers.SGD(lr=0.02, momentum=0.0, decay=0.0, nesterov=False),
              loss="categorical_crossentropy",
              metrics=['accuracy'])

# train model
train_log = dict([("acc", []), ("loss", [])])
test_log = dict([("acc", []), ("loss", [])])

for i in range(0, TRAINING_ITERATIONS, 1):
    print("+++ ITERATION {}/{} +++".format(i, TRAINING_ITERATIONS))
    model.fit([X_q_train, X_c_train], Y_train, epochs=5, batch_size=32)

    print("+++ Evaluate train set +++")
    score = model.evaluate([X_q_train, X_c_train], Y_train, batch_size=64)
    train_log["acc"].append(score[1])
    train_log["loss"].append(score[0])
    print("+++ Evaluate test set +++")
    score = model.evaluate([X_q_test, X_c_test], Y_test, batch_size=64)
    test_log["acc"].append(score[1])
    test_log["loss"].append(score[0])

# plot learning process
x = np.arange(0, len(test_log["acc"]), step=1)
plt.plot(x, test_log["loss"], 'r--', x, test_log["acc"], 'r', x, train_log["loss"], 'g--', x, train_log["acc"], 'g')
plt.title("Feedforward")
plt.savefig("plot.png")
