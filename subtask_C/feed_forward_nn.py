from subtask_C.data_set import word2vec_dataset
import word2vec_model.word2vec_utils as word2vec
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
#from keras.layers import Dense, Dropout

#load X, y
model = word2vec.load_word2vec_model("SemEval2016-Task3-CQA-QL-dev_model")
org_questions, rel_comments, y = word2vec_dataset("../data/SemEval2016-Task3-CQA-QL-dev.xml", model)
X = np.concatenate((org_questions, rel_comments), axis=1)

#transform class labels
lb = LabelBinarizer()
y = lb.fit_transform(y)

# split into test and train sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)

for array in (X_train, X_test, Y_train, Y_test):
    print(array.shape)