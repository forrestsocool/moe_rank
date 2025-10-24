import numpy as np
from math import sqrt
from collections import OrderedDict, namedtuple, defaultdict
from sklearn.metrics import roc_auc_score, log_loss

names = namedtuple('names', ['LOGLOSS', 'AUC', 'gAUC', 'RMSE'])
names.__new__.__defaults__ = ('logloss', 'auc', 'gauc', 'rmse')


def get_metrics(metrics_name):
    metrics = OrderedDict()
    if isinstance(metrics_name, tuple):
        for metric_name in metrics_name:
            if str(metric_name).lower() == "binary_crossentropy" or str(metric_name).lower() == "logloss":
                metrics[metric_name] = log_loss
            if str(metric_name).lower() == "auc":
                metrics[metric_name] = roc_auc_score
            if str(metric_name).lower() == "gauc":
                metrics[metric_name] = gAUC
            if str(metric_name).lower() == "rmse":
                metrics[metric_name] = rmse
    else:
        raise ValueError("The name of metrics should be tuple")
    return metrics


def gAUC(labels, preds, user_ids):
    group_score = defaultdict(lambda:[])
    group_truth = defaultdict(lambda:[])
    for idx, truth in enumerate(labels):
        tempId = user_ids[idx]
        score = preds[idx]
        truth = labels[idx]
        group_score[tempId].append(score)
        group_truth[tempId].append(truth)
        group_flag = defaultdict(lambda: False)
    for tempId in set(user_ids):
        truths = group_truth[tempId]
        label_sum = sum(truths)
        if 0 < label_sum < len(truths):
            group_flag[tempId] = True
    impression_total = 0
    total_auc = 0
    for tempid in group_flag:
        if group_flag[tempid]:
            auc = roc_auc_score(group_truth[tempid], group_score[tempid])
            if auc > 0.0:
                total_auc += auc * len(group_truth[tempid])
                impression_total += len(group_truth[tempid])
    group_auc = float(total_auc) / impression_total
    return group_auc


def rmse(score_label, pred_scores, acc_set, avg_loss_set, squared_error):
    avg_loss_set.append(np.mean(np.abs(pred_scores - score_label)))
    squared_error.extend(np.abs(pred_scores - score_label) ** 2)

    diff = np.abs(pred_scores - score_label)
    diff[diff > 0.5] = 1
    acc = 1 - np.mean(diff)
    acc_set.append(acc)
    rmse = sqrt(np.sum(squared_error) / len(squared_error))

    return np.mean(acc_set), np.mean(avg_loss_set), rmse
