from subtask_C.data_set import word2vec_dataset
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout
from subtask_C.custom_metrics import precision, recall
import matplotlib.pyplot as plt
import word2vec_model.word2vec_utils as word2vec
import numpy as np
import keras


TRAINING_ITERATIONS = 10


def create_model(units=[1], activ=["elu"], init=[keras.initializers.he_normal()], input_dim=200):
    model = keras.Sequential()
    model.add(Dense(units=units[0], activation=activ[0], kernel_initializer=init[0], input_dim=input_dim))
    if len(units) > 1:
        for current_layer_units, current_layer_activ, current_layer_init in zip(units[1:], activ[1:], init[1:]):
            model.add(Dense(units=current_layer_units, activation=current_layer_activ, kernel_initializer=current_layer_init))
            model.add(Dropout(0.5))
    model.add(Dense(3, activation="softmax"))
    
    model.compile(optimizer=keras.optimizers.SGD(lr=0.03, momentum=0.0, decay=0.0, nesterov=False),
              loss="categorical_crossentropy",
              metrics=['accuracy'])
    return model


#load X, y
word2vec_model = word2vec.load_word2vec_model("SemEval2016-Task3-CQA-QL-dev_model")
org_questions, rel_comments, y = word2vec_dataset("data/SemEval2016-Task3-CQA-QL-dev.xml", word2vec_model)
X = np.concatenate((org_questions, rel_comments), axis=1)

#transform class labels
lb = LabelBinarizer()
y = lb.fit_transform(y)

# split into test and train sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#create compiled model
units = [50]
model = create_model(units=units, input_dim=X_train.shape[1])

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
test_loss, test_acc, train_loss, train_acc = plt.plot(x, test_log["loss"], 'r--', x, test_log["acc"], 'r', x, train_log["loss"], 'g--', x, train_log["acc"], 'g')
test_loss.set_label("loss (test)")
test_acc.set_label("acc (test)")
train_loss.set_label("loss (train)")
train_acc.set_label("acc (train)")
plt.legend(handles=[test_loss, test_acc, train_loss, train_acc], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title("Feedforward")
plt.savefig("subtask_C\plot.png", bbox_inches = "tight")