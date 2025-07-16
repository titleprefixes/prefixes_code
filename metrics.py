import numpy as np


def rank_predictions_metrics(predictions, y, k):
    hits_scores = []
    for i, probabilities in enumerate(predictions):
        top_classes = (-probabilities).argsort(kind='mergesort')[:k]  # mergesort for stability of tie breaks
        true_class = y[i]
        tag = 0
        if true_class in top_classes:
            tag = 1
        hits_scores.append(tag)
    return hits_scores, np.mean(hits_scores)


def recall(y_pred, y, label=0):
    indices = np.where(y == label)[0]
    projected_pred = y_pred[indices]
    projected_y = y[indices]
    return accuracy(projected_pred, projected_y)


def precision(y_pred, y, label=0):
    indices = np.where(y_pred == label)[0]
    projected_pred = y_pred[indices]
    projected_y = y[indices]
    return accuracy(projected_pred, projected_y)


def accuracy(y_pred, y):
    if len(y) == 0:
        return 0
    return sum([1 for a, b in zip(y_pred, y) if a == b]) / len(y)


def get_all_classes_stats(fn, y_pred, y):
    labels = list(set(y))
    results = {}
    for label in labels:
        results[label] = fn(y_pred, y, label)
    return results
