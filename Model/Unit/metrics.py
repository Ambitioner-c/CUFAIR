# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/8/18 15:30
import numpy as np
from scipy.special import softmax

from sklearn import metrics


def confusion_matrix(input: np.array, target: np.array) -> [int]:
    y_true = target
    y_pred = np.argmax(input, axis=1)

    return metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)


def accuracy_score(input: np.array, target: np.array) -> float:
    y_true = target
    y_pred = np.argmax(input, axis=1)

    return metrics.accuracy_score(y_true=y_true, y_pred=y_pred)


def precision_score(input: np.array, target: np.array, average: str='weighted') -> float:
    y_true = target
    y_pred = np.argmax(input, axis=1)

    return metrics.precision_score(y_true=y_true, y_pred=y_pred, average=average)


def recall_score(input: np.array, target: np.array, average: str='weighted') -> float:
    y_true = target
    y_pred = np.argmax(input, axis=1)

    return metrics.recall_score(y_true=y_true, y_pred=y_pred, average=average)


def f1_score(input: np.array, target: np.array, average: str='weighted') -> float:
    y_true = target
    y_pred = np.argmax(input, axis=1)

    return metrics.f1_score(y_true=y_true, y_pred=y_pred, average=average)


def roc_auc_score(input: np.array, target: np.array, average: str='weighted', multi_class: str='ovr') -> float:
    y_true = target
    y_pred = softmax(input, axis=1)

    return metrics.roc_auc_score(y_true=y_true, y_score=y_pred, average=average, multi_class=multi_class)


def mean_absolute_error():
    pass


def mean_squared_error():
    pass


def root_mean_squared_error():
    pass


def r2_score():
    pass


def main():
    # y_pred = [0, 2, 1, 0, 0, 1]
    # y_true = [0, 1, 2, 0, 1, 2]
    input = np.array([[2, 1, 0],
                      [0, 1, 2],
                      [1, 2, 0],
                      [2, 0, 1],
                      [2, 1, 0],
                      [0, 2, 1]])
    target = np.array([0, 1, 2, 0, 1, 2])

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


if __name__ == '__main__':
    main()
