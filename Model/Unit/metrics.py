# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/8/18 15:30
import math
import numpy as np
from scipy.special import softmax

from sklearn import metrics


def confusion_matrix(input: np.array, target: np.array) -> [int]:
    """
    :param input:
    :param target:
    :return: [tn, fp, fn, tp]
    """
    y_true = target
    y_pred = np.argmax(input, axis=1)

    return metrics.confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()


def accuracy_score(input: np.array, target: np.array) -> float:
    y_true = target
    y_pred = np.argmax(input, axis=1)

    return metrics.accuracy_score(y_true=y_true, y_pred=y_pred)


def precision_score(input: np.array, target: np.array, average: str = 'weighted') -> float:
    y_true = target
    y_pred = np.argmax(input, axis=1)

    return metrics.precision_score(y_true=y_true, y_pred=y_pred, average=average)


def recall_score(input: np.array, target: np.array, average: str = 'weighted') -> float:
    y_true = target
    y_pred = np.argmax(input, axis=1)

    return metrics.recall_score(y_true=y_true, y_pred=y_pred, average=average)


def f1_score(input: np.array, target: np.array, average: str = 'weighted') -> float:
    y_true = target
    y_pred = np.argmax(input, axis=1)

    return metrics.f1_score(y_true=y_true, y_pred=y_pred, average=average)


def roc_auc_score(input: np.array, target: np.array, average: str = 'weighted') -> float:
    y_true = target
    y_pred = softmax(input, axis=1)[:, 1]

    return metrics.roc_auc_score(y_true=y_true, y_score=y_pred, average=average)

def sort_and_couple(labels: np.array, scores: np.array) -> np.array:
    """Zip the `labels` with `scores` into a single list."""
    couple = list(zip(labels, scores))
    return np.array(sorted(couple, key=lambda x: x[1], reverse=True))

def precision(input: np.array, target: np.array, k: int = 1, threshold: float = 0.) -> float:
    y_true = target
    y_pred = input

    coupled_pair = sort_and_couple(y_true, y_pred)
    p = 0.0
    for idx, (label, score) in enumerate(coupled_pair):
        if idx >= k:
            break
        if label > threshold:
            p += 1.

    return p / k

def average_precision(input: np.array, target: np.array, threshold: float = 0.) -> float:
    out = [precision(input, target, k + 1, threshold) for k in range(len(input))]
    if not out:
        return 0.

    return np.mean(out).item()

def discounted_cumulative_gain(input: np.array, target: np.array, k: int = 1, threshold: float = 0.) -> float:
    y_true = target
    y_pred = input

    if k <= 0:
        return 0.
    coupled_pair = sort_and_couple(y_true, y_pred)
    result = 0.
    for idx, (label, score) in enumerate(coupled_pair):
        if idx >= k:
            break
        if label > threshold:
            result += (math.pow(2., label) - 1.) / math.log(2. + idx)
    return result

def normalized_discounted_cumulative_gain(input: np.array, target: np.array, k: int = 1, threshold: float = 0.) -> float:
    y_true = target
    y_pred = input

    idcg_val = discounted_cumulative_gain(input=y_true, target=y_true, k=k, threshold=threshold)
    dcg_val = discounted_cumulative_gain(input=y_pred, target=y_true, k=k, threshold=threshold)
    return dcg_val / idcg_val if idcg_val != 0 else 0

def mean_average_precision(input: np.array, target: np.array, threshold: float = 0.) -> float:
    y_true = target
    y_pred = input

    result = 0.
    pos = 0
    coupled_pair = sort_and_couple(y_true, y_pred)
    for idx, (label, score) in enumerate(coupled_pair):
        if label > threshold:
            pos += 1.
            result += pos / (idx + 1.)
    if pos == 0:
        return 0.
    else:
        return result / pos

def mean_reciprocal_rank(input: np.array, target: np.array, threshold: float = 0.) -> float:
    y_true = target
    y_pred = input

    coupled_pair = sort_and_couple(y_true, y_pred)
    for idx, (label, pred) in enumerate(coupled_pair):
        if label > threshold:
            return 1. / (idx + 1)
    return 0.

def classification():
    # y_pred = [0, 2, 1, 0, 0, 1]
    # y_true = [0, 1, 2, 0, 1, 2]
    input = np.array([[2, 1],
                      [0, 1],
                      [1, 2],
                      [2, 0],
                      [2, 1],
                      [0, 2]])
    target = np.array([0, 1, 0, 0, 1, 1])

    cm = confusion_matrix(input, target)
    print(f'Confusion Matrix: {cm}')

    acc = accuracy_score(input, target)
    print(f'Accuracy: {acc}')

    pre = precision_score(input, target)
    print(f'Precision: {pre}')

    rec = recall_score(input, target)
    print(f'Recall: {rec}')

    f1 = f1_score(input, target)
    print(f'F1: {f1}')

    roc_auc = roc_auc_score(input, target)
    print(f'ROC AUC: {roc_auc}')


def ranking():
    # y_true = [0, 0, 0, 1]
    # y_pred = [0.2, 0.4, 0.3, 0.1]
    input = np.array([0.2, 0.4, 0.3, 0.1])
    target = np.array([0, 0, 0, 1])

    p = precision(input, target, k=4)
    print(f'Precision@1: {p}')

    ap = average_precision([0.1, 0.6], [0, 1])
    print(f'Average Precision: {ap}')

    dcg = discounted_cumulative_gain([0.4, 0.2, 0.5, 0.7], [0, 1, 2, 0], k=2)
    print(f'DCG@2: {dcg}')

    ndcg = normalized_discounted_cumulative_gain([0.4, 0.2, 0.5, 0.7], [0, 1, 2, 0], k=2)
    print(f'NDCG@2: {ndcg}')

    map = mean_average_precision([0.1, 0.6, 0.2, 0.3], [0, 1, 0, 0])
    print(f'MAP: {map}')

    mrr = mean_reciprocal_rank([0.2, 0.3, 0.7, 1.0], [1, 0, 0, 0])
    print(f'MRR: {mrr}')


if __name__ == '__main__':
    ranking()
