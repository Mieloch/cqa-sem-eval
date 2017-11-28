from subtask_C.data_set import sentences_dataset
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout
from subtask_C.custom_metrics import precision, recall
import matplotlib.pyplot as plt
import word2vec_model.word2vec_utils as word2vec
import numpy as np
import keras


TRAINING_ITERATIONS = 100


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
              metrics=[precision, recall])
    return model


#load X, y
word2vec_model = word2vec.load_word2vec_model("SemEval2016-Task3-CQA-QL-dev_model")
org_questions, rel_comments, y = sentences_dataset("data/SemEval2016-Task3-CQA-QL-dev.xml", word2vec_model)
X = np.concatenate((org_questions, rel_comments), axis=1)

#transform class labels
lb = LabelBinarizer()
y = lb.fit_transform(y)

# split into test and train sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#create compiled model
units = [50, 100, 100]
activ=["elu", "elu", "elu"]
init=[keras.initializers.he_normal(), keras.initializers.he_normal(), keras.initializers.he_normal()]
model = create_model(units=units, init=init, activ=activ, input_dim=X_train.shape[1])

# train model
train_log = dict([("prec", []), ("rec", []), ("loss", [])])
test_log = dict([("prec", []), ("rec", []), ("loss", [])])

for i in range(0, TRAINING_ITERATIONS, 1):
    print("+++ ITERATION {}/{} +++".format(i, TRAINING_ITERATIONS))
    model.fit(X_train, Y_train, epochs=5, batch_size=32)

    print("+++ Evaluate train set +++")
    score = model.evaluate(X_train, Y_train, batch_size=64)
    train_log["prec"].append(score[1])
    train_log["rec"].append(score[2])
    train_log["loss"].append(score[0])
    print("+++ Evaluate test set +++")
    score = model.evaluate(X_test, Y_test, batch_size=64)
    test_log["prec"].append(score[1])
    test_log["rec"].append(score[2])
    test_log["loss"].append(score[0])

# plot learning process
x = np.arange(0, TRAINING_ITERATIONS, step=1)

plt.subplot(211)
train_loss, train_prec, train_rec = plt.plot(x, train_log["loss"], 'r', x, train_log["prec"], 'g', x, train_log["rec"], 'b')
train_loss.set_label("loss (train)")
train_prec.set_label("precision (train)")
train_rec.set_label("recall (train)")
plt.legend(handles=[train_loss, train_prec, train_rec], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title("Feedforward - train")

plt.subplot(212)
test_loss, test_prec, test_rec = plt.plot(x, test_log["loss"], 'r', x, test_log["prec"], 'g', x, test_log["rec"], 'b')
test_loss.set_label("loss (test)")
test_prec.set_label("precision (test)")
test_rec.set_label("recall (test)")
plt.legend(handles=[test_loss, test_prec, test_rec], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title("Feedforward - test")

plt.savefig("subtask_C\plot.png", bbox_inches = "tight")