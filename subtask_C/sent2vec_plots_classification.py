import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
import csv
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def make_meshgrid(x, y, h=.05):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def string_to_class(text):
    if text == "Good":
        return 0
    if text == "PotentiallyUseful":
        return 1
    if text == "Bad":
        return 2


with open('OrgQuestion_to_RelComment_stats.csv') as csvfile:
    # load data
    Y = []
    X = []
    reader = csv.DictReader(csvfile)
    for row in reader:
        X.append((float(row["w2v_cosine_similarity"])))
        relevance_ = row["relevance"]
        Y.append(relevance_)
    Y = list(map(string_to_class, Y))

    X = np.array(X)
    Y = np.array(Y)

    # split data to train and test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    X_train = X_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)
    # Y_train = Y_train.reshape(-1, 1)
    # Y_test = Y_test.reshape(-1, 1)

    # classify
    clf = LogisticRegression(multi_class="multinomial", solver="lbfgs")
    clf.fit(X_train, Y_train)
    # evaluate
    print("Accuracy score =", accuracy_score(Y_test, clf.predict(X_test)))

    # prepare contour mesh grid and colors
    x_axis = np.arange(0, len(X_test), 1)
    y_axis = np.arange(0, 1, 0.05)
    xx, yy = make_meshgrid(x_axis, y_axis)
    Z = clf.predict(np.ravel(yy).reshape(-1, 1))
    Z = Z.reshape(yy.shape)

    # plots
    plt.ylabel('cosine similarity')
    plt.xlabel('n-th sample')
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
    plt.scatter(x_axis, X_test, c=Y_test)
    irrelevant_label = mpatches.Patch(color=(0.5, 0.0, 0.0), label='Bad')
    relevant_label = mpatches.Patch(color=(0.5, 1, 0.5), label='PotentiallyUseful')
    perfect_match_label = mpatches.Patch(color=(0.0, 0.0, 0.5), label='Good')
    plt.legend(handles=[irrelevant_label, relevant_label, perfect_match_label])
    plt.show()
