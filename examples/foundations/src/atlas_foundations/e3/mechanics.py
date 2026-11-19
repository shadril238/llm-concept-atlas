"""Transparent foundation calculations. No learned-model claims."""

from math import cos, exp, isfinite, log, sin, sqrt
from random import Random


def softmax(logits, temperature=1.0):
    if (
        not logits
        or not isfinite(temperature)
        or temperature <= 0
        or not all(isfinite(x) for x in logits)
    ):
        raise ValueError(
            "Finite nonempty logits and a positive temperature are required"
        )
    shifted = [exp((x - max(logits)) / temperature) for x in logits]
    return [x / sum(shifted) for x in shifted]


def filter_distribution(logits, temperature=1.0, top_k=None, top_p=1.0):
    if not 0 < top_p <= 1 or (top_k is not None and not 1 <= top_k <= len(logits)):
        raise ValueError("Use 0 < top_p <= 1 and a valid top_k")
    probabilities = softmax(logits, temperature)
    order = sorted(range(len(logits)), key=lambda i: (-probabilities[i], i))
    order = order[:top_k] if top_k else order
    denominator = sum(probabilities[i] for i in order)
    retained = []
    cumulative = 0.0
    for i in order:
        retained.append(i)
        cumulative += probabilities[i] / denominator
        if cumulative >= top_p:
            break
    total = sum(probabilities[i] for i in retained)
    return [
        probabilities[i] / total if i in retained else 0.0 for i in range(len(logits))
    ]


def sample(probabilities, seed):
    return Random(seed).choices(range(len(probabilities)), weights=probabilities, k=1)[
        0
    ]


def attention(queries, keys, values, causal=True, segments=None):
    if not queries or not keys or len(keys) != len(values):
        raise ValueError("Nonempty compatible Q/K/V required")
    width = len(queries[0])
    if (
        not width
        or not values[0]
        or any(len(row) != width for row in queries + keys)
        or any(len(row) != len(values[0]) for row in values)
    ):
        raise ValueError("Q/K widths and V widths must be consistent")
    if any(not isfinite(x) for row in queries + keys + values for x in row):
        raise ValueError("Finite Q/K/V required")
    if segments is not None and (
        len(segments) != len(keys) or len(queries) != len(keys)
    ):
        raise ValueError(
            "Segment mask requires one segment per self-attention position"
        )
    weights = []
    outputs = []
    for i, query in enumerate(queries):
        allowed = [
            j
            for j in range(len(keys))
            if (not causal or j <= i)
            and (segments is None or segments[i] == segments[j])
        ]
        scores = [
            sum(a * b for a, b in zip(query, keys[j])) / sqrt(width) for j in allowed
        ]
        p = softmax(scores)
        row = [0.0] * len(keys)
        for j, value in zip(allowed, p):
            row[j] = value
        weights.append(row)
        outputs.append(
            [
                sum(row[j] * values[j][d] for j in range(len(keys)))
                for d in range(len(values[0]))
            ]
        )
    return weights, outputs


def rope_pair(vector, position, frequency=1.0):
    if len(vector) != 2:
        raise ValueError("One coordinate pair is required")
    angle = position * frequency
    a, b = vector
    return [a * cos(angle) - b * sin(angle), a * sin(angle) + b * cos(angle)]


def normalize(vector, rms=False, epsilon=1e-5):
    if not vector or epsilon <= 0 or not all(isfinite(x) for x in vector):
        raise ValueError("Finite vector and positive epsilon required")
    mean = 0.0 if rms else sum(vector) / len(vector)
    scale = sqrt(sum((x - mean) ** 2 for x in vector) / len(vector) + epsilon)
    return [(x - mean) / scale for x in vector]


def cosine(a, b):
    denominator = sqrt(sum(x * x for x in a) * sum(x * x for x in b))
    if not denominator:
        raise ValueError("Cosine is undefined for a zero vector")
    return sum(x * y for x, y in zip(a, b)) / denominator


def pairs(items, window=1):
    return [
        (center, items[j])
        for i, center in enumerate(items)
        for j in range(max(0, i - window), min(len(items), i + window + 1))
        if i != j
    ]


def kv_bytes(layers, batch, heads, positions, head_width, bytes_per_value=2):
    return 2 * layers * batch * heads * positions * head_width * bytes_per_value


def negative_loss(center, positive, negative):
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))
    # log(1 + exp(x)) is a stable softplus, avoiding log(0).
    softplus = lambda x: max(x, 0) + log(1 + exp(-abs(x)))
    return softplus(-dot(center, positive)) + softplus(dot(center, negative))


def main():
    q = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    v = [[1.0, 0.0], [0.0, 2.0], [3.0, 1.0]]
    weights, output = attention(q, q, v)
    assert all(weights[i][j] == 0 for i in range(3) for j in range(i + 1, 3))
    distribution = filter_distribution([2.0, 1.0, 0.0], top_p=0.8)
    assert abs(sum(distribution) - 1) < 1e-12
    print("causal weights:", [[round(x, 4) for x in row] for row in weights])
    print("last output:", [round(x, 4) for x in output[-1]])
    print("top-p probabilities:", [round(x, 4) for x in distribution])
    print("rotated pair:", [round(x, 4) for x in rope_pair([1.0, 0.0], 1)])
    print("cache bytes:", kv_bytes(4, 1, 2, 10, 8))


if __name__ == "__main__":
    main()
