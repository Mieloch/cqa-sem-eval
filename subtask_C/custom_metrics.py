import keras.backend as K
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import csv


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / predicted_positives
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / possible_positives
    return recall


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.values_f1 = []
        self.values_recall = []
        self.values_precision = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
        val_targ = self.model.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.values_f1.append(_val_f1)
        self.values_recall.append(_val_recall)
        self.values_precision.append(_val_precision)
        print(" — f1: " + _val_f1 + " — precision: " + _val_precision + " — recall: " + _val_recall)
        return


def save_params_csv(filename, params_dict):
    with open(filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[key for key in params_dict])
        writer.writeheader()
        writer.writerow(params_dict)
        csvfile.close()
    return

def get_acc_plot(_plt, log, subplotn):
    x = np.arange(0, len(log["acc"]), step=1)
    _plt.subplot(subplotn)
    _loss, _acc = _plt.plot(x, log["loss"], 'r', x, log["acc"], 'b')
    _loss.set_label("loss")
    _acc.set_label("acc")
    _plt.legend(handles=[_loss, _acc], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    return _plt

