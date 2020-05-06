import numpy as np
from sklearn import metrics

def f1_score(preds, targets, sigmoid=True, thresh=0.5, average='micro', idx=None):
    if sigmoid: preds = 1/(1 + np.exp(-preds))
    preds = (preds >= thresh).astype(np.uint8)
    if idx is not None:
        return metrics.fbeta_score(y_true=targets, y_pred=preds, beta=1, average=None)[idx]
    return metrics.fbeta_score(y_true=targets, y_pred=preds, beta=1, average=average)

def create_metrics_DISEASE(func, name, label_cols_list):
    return lambda preds, target: func(preds, targets, idx=label_cols_list.index(name))

def classification_counts(y_pred, y_true, log=False):
    TP,FP,TN,FN = 0,0,0,0

    for i in range(len(y_pred)): 
        if y_true[i]==y_pred[i]==1:
            TP += 1
        if y_pred[i]==1 and y_true[i]== 0:
            FP += 1
        if y_true[i]==y_pred[i]==0:
            TN += 1
        if y_pred[i]==0 and y_true[i]==1:
            FN += 1
    if log:
        print('TP\tFP\tTN\tFN\t')
        print(f'{TP}\t{FP}\t{TN}\t{FN}\t')
    return {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}

def binary_metrics(y_pred, y_true, log=False):
    precision = metrics.precision_score(y_true, y_pred, average='binary')
    recall = metrics.recall_score(y_true, y_pred, average='binary')
    spec = metrics.recall_score(y_true, y_pred, average='binary', pos_label=0)
    f1 = metrics.f1_score(y_true, y_pred, average='binary')
    f2 = metrics.fbeta_score(y_true, y_pred, average='binary', beta=2)
    auc = metrics.roc_auc_score(y_true, y_pred)
    acc = metrics.accuracy_score(y_true, y_pred)
    if log:
        print('pre\trecall\tspec\tacc\tauc\tf2\tf1\t')
        print(f'{precision:0.4f}\t{recall:0.4f}\t{spec:0.4f}\t{acc:0.4f}\t{auc:0.4f}\t{f2:0.4f}\t{f1:0.4f}')
    return {
        'precision': precision, 'recall': recall, 'specificity': spec, 
        'accuracy': acc, 'auc': auc, 'f2': f2, 'f1': f1 
    }