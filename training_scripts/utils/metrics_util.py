from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import matthews_corrcoef
import numpy as np

def get_all_metrics(model, eval_data, eval_labels, pred_labels):
    (fpr, tpr, thresholds_keras) = roc_curve(eval_labels, pred_labels)
    auc_ = auc(fpr, tpr)
    print('auc_keras: {:.3f}'.format(auc_))
    correct_predictions = np.equal(eval_labels, pred_labels)
    accuracy = np.mean(correct_predictions.astype(float))
    print('Test accuracy: {:.3f}'.format(accuracy))
    precision = precision_score(eval_labels, pred_labels, zero_division=0)
    print('Precision score: {:.3f}'.format(precision))
    recall = recall_score(eval_labels, pred_labels)
    print('Recall score: {:.3f}'.format(recall))
    f1 = f1_score(eval_labels, pred_labels)
    print('F1 score: {:.3f}'.format(f1))
    mcc = matthews_corrcoef(eval_labels, pred_labels)
    print('MCC: {:.3f}'.format(mcc))
    average_precision = average_precision_score(eval_labels, pred_labels)
    print('Average precision-recall score: {:.3f}'.format(average_precision))
    return (auc_, accuracy, precision, recall, f1, average_precision, fpr, tpr, mcc)

def get_all_metrics_(eval_labels, pred_labels):
    (fpr, tpr, thresholds_keras) = roc_curve(eval_labels, pred_labels)
    auc_ = auc(fpr, tpr)
    print('auc_keras:' + str(auc_))
    precision = precision_score(eval_labels, pred_labels, zero_division=0)
    print('Precision score: {0:0.2f}'.format(precision))
    recall = recall_score(eval_labels, pred_labels)
    print('Recall score: {0:0.2f}'.format(recall))
    f1 = f1_score(eval_labels, pred_labels)
    print('F1 score: {0:0.2f}'.format(f1))
    mcc = matthews_corrcoef(eval_labels, pred_labels)
    print('MCC: {0:0.2f}'.format(mcc))
    average_precision = average_precision_score(eval_labels, pred_labels)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))
    return (auc_, precision, recall, f1, average_precision, fpr, tpr, mcc)