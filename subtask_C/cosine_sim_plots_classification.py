import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
import csv
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
GOOD_COLOR = (0.0, 0.0, 0.5)

USEFUL_COLOR = (0.5, 1, 0.5)

BAD_COLOR = (0.5, 0.0, 0.0)


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
    # sort test test
    test_set = list(zip(X_test, Y_test))
    test_set.sort(key=lambda d: d[1])
    X_test = np.array([i[0] for i in test_set])
    Y_test = np.array([i[1] for i in test_set])

    X_train = X_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)
    Y_train = Y_train.reshape(-1, 1)
    Y_test = Y_test.reshape(-1, 1)

    # classify
    clf = LogisticRegression(multi_class="multinomial", solver="lbfgs")
    clf.fit(X_train, Y_train)
    # evaluate
    print("Test set accuracy score =", accuracy_score(Y_test, clf.predict(X_test)))
    print("Train set accuracy score =", accuracy_score(Y_train, clf.predict(X_train)))
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
    irrelevant_label = mpatches.Patch(color=BAD_COLOR, label='Bad')
    relevant_label = mpatches.Patch(color=USEFUL_COLOR, label='PotentiallyUseful')
    perfect_match_label = mpatches.Patch(color=GOOD_COLOR, label='Good')
    plt.legend(handles=[irrelevant_label, relevant_label, perfect_match_label])
    plt.show()
