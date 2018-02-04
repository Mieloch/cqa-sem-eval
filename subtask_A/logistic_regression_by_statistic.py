import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

RELEVANCE = "relevance"
LENGTH_DIFFERENCE = "length_difference"
JACCARD_DISTANCE = "jaccard_distance"
COSINE_SIMILARITY = "cosine_similarity"
STATS_FILE = "csv/statistic_metrics.csv"

data_set = pd.read_csv(STATS_FILE)
by_stat = data_set[[JACCARD_DISTANCE, COSINE_SIMILARITY, LENGTH_DIFFERENCE, RELEVANCE]].values
Y = by_stat[:, 3].reshape(-1, 1)
X = by_stat[:, 0:3]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

clf = LogisticRegression()
clf.fit(X_train, Y_train)

print("Train set accuracy score =", accuracy_score(Y_train, clf.predict(X_train)))
print("Train set recall score =", recall_score(Y_train, clf.predict(X_train)))
print("Train set precission score =", precision_score(Y_train, clf.predict(X_train)))
print("Test set accuracy score =", accuracy_score(Y_test, clf.predict(X_test)))
print("Test set recall score =", recall_score(Y_test, clf.predict(X_test)))
print("Test set precission score =", precision_score(Y_test, clf.predict(X_test)))
