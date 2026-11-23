"""Stable symmetric contrastive loss from a square, paired logit matrix."""

import math


def contrastive_loss(logits):
    n = len(logits)
    if (
        n < 2
        or any(len(row) != n for row in logits)
        or not all(math.isfinite(v) for row in logits for v in row)
    ):
        raise ValueError("Finite square paired matrix with at least two pairs required")

    def direction(rows):
        terms = []
        for i, row in enumerate(rows):
            maximum = max(row)
            logsum = maximum + math.log(sum(math.exp(v - maximum) for v in row))
            terms.append(logsum - row[i])
        return terms

    image = direction(logits)
    text = direction(list(map(list, zip(*logits))))
    return {
        "image_to_text": sum(image) / n,
        "text_to_image": sum(text) / n,
        "loss": (sum(image) + sum(text)) / (2 * n),
        "image_terms": image,
        "text_terms": text,
    }
