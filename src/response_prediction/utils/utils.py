import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, balanced_accuracy_score, precision_recall_curve, auc, average_precision_score
from sklearn import metrics
import logging

def get_AUC(predict, label, ignore_idx):
    n = np.shape(label)[1]
    auc = np.zeros([n,], dtype=np.float32)
    for i in range(n):
        idx = ~ignore_idx[:,i]
        auc[i] = roc_auc_score(label[idx,i], predict[idx,i])
    return auc

def get_Accuracy(predict, label, ignore_idx):
    n = np.shape(label)[1]
    accuracy = np.zeros([n,], dtype=np.float32)
    for i in range(n):
        acc = 0.0
        for j in range(0,np.shape(label)[0]):
            if ignore_idx[j,i]:
                continue
            else:
                if predict[j,i] == label[j,i]:
                    acc += 1.0
        accuracy[i] = acc/np.sum(~ignore_idx[:,i])
    return accuracy


def get_AUPR(predict, label, ignore_idx):
    n = np.shape(label)[1]
    aupr = np.zeros([n,], dtype=np.float32)
    for i in range(n):
        idx = ~ignore_idx[:,i]
        aupr[i] = average_precision_score(label[idx,i], predict[idx,i])
    return aupr

def get_Fscore(predict, label, ignore_idx):
    n = np.shape(label)[1]
    precision = np.zeros([n,], dtype=np.float32)
    recall = np.zeros([n,], dtype=np.float32)
    fscore = np.zeros([n,], dtype=np.float32)
    for i in range(n):
        idx = ~ignore_idx[:,i]
        score = precision_recall_fscore_support(label[idx,i], predict[idx,i], pos_label=1, beta=1.0, average='binary')
        precision[i] = score[0]
        recall[i] = score[1]
        fscore[i] = score[2]
    return [precision, recall, fscore]

def get_performance(label, predict_prob, predict, ignore_idx):
    prec_rec_fscore = get_Fscore(predict, label, ignore_idx)
    return [
        get_Accuracy(predict, label, ignore_idx),
        get_AUC(predict_prob, label, ignore_idx),
        get_AUPR(predict_prob, label, ignore_idx),
        prec_rec_fscore[0],
        prec_rec_fscore[1],
        prec_rec_fscore[2]
    ]
    
class TqdmLoggingHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, end=self.terminator)
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)
