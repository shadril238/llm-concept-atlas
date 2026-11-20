"""Explicit binary metrics; rejected/invalid predictions remain in denominators."""

import math


def parse_label(text):
    if not isinstance(text, str):
        return -1
    return {"negative": 0, "positive": 1}.get(text.strip().casefold(), -1)


def classification_metrics(gold, predicted):
    if not gold or len(gold) != len(predicted):
        raise ValueError("Nonempty aligned labels are required")
    if any(type(v) is not int or v not in [0, 1] for v in gold):
        raise ValueError("Gold must use integer label IDs 0/1")
    if any(type(v) is not int or v not in [-1, 0, 1] for v in predicted):
        raise ValueError("Predictions must use 0/1 or invalid=-1")
    matrix = [[0, 0, 0], [0, 0, 0]]
    for actual, prediction in zip(gold, predicted):
        matrix[actual][prediction if prediction >= 0 else 2] += 1
    per_class = []
    for label in [0, 1]:
        tp = matrix[label][label]
        fp = matrix[1 - label][label]
        fn = sum(matrix[label]) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        per_class.append(
            dict(
                label=label,
                support=sum(matrix[label]),
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )
    return {
        "count": len(gold),
        "confusion": matrix,
        "accuracy": sum(matrix[i][i] for i in [0, 1]) / len(gold),
        "macro_f1": sum(row["f1"] for row in per_class) / 2,
        "weighted_f1": sum(row["f1"] * row["support"] for row in per_class) / len(gold),
        "invalid_rate": sum(row[2] for row in matrix) / len(gold),
        "per_class": per_class,
    }


def selective_metrics(gold, predicted, confidences, threshold):
    if (
        len(confidences) != len(gold)
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("Aligned scores and a threshold in [0,1] are required")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in confidences):
        raise ValueError("Scores must be finite in [0,1]")
    classification_metrics(gold, predicted)
    accepted = [
        i
        for i, (prediction, score) in enumerate(zip(predicted, confidences))
        if prediction >= 0 and score >= threshold
    ]
    return {
        "accepted": len(accepted),
        "review": len(gold) - len(accepted),
        "coverage": len(accepted) / len(gold),
        "accepted_accuracy": sum(gold[i] == predicted[i] for i in accepted)
        / len(accepted)
        if accepted
        else None,
    }
