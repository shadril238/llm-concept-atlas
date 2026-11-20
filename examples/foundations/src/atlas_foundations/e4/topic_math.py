"""Hand-worked convention matching BERTopic 0.17.4's default c-TF-IDF."""

from math import isfinite, log, sqrt


def class_tfidf(counts):
    if not counts or not counts[0] or any(len(r) != len(counts[0]) for r in counts):
        raise ValueError("A nonempty rectangular count table is required")
    if any(not isfinite(v) or v < 0 for row in counts for v in row):
        raise ValueError("Counts must be finite and nonnegative")
    totals = [sum(row) for row in counts]
    if any(total <= 0 for total in totals):
        raise ValueError("Each class must contain a token")
    frequencies = [sum(row[j] for row in counts) for j in range(len(counts[0]))]
    if any(value <= 0 for value in frequencies):
        raise ValueError("Vocabulary columns must occur")
    average = int(sum(totals) / len(totals))
    idf = [log(1 + average / frequency) for frequency in frequencies]
    tf = [[count / total for count in row] for row, total in zip(counts, totals)]
    return {
        "average_length": average,
        "frequencies": frequencies,
        "idf": idf,
        "tf": tf,
        "weights": [
            [frequency * weight for frequency, weight in zip(row, idf)] for row in tf
        ],
    }


def cosine(a, b):
    if not a or len(a) != len(b) or any(not isfinite(v) for v in [*a, *b]):
        raise ValueError("Finite aligned vectors are required")
    norm = sqrt(sum(v * v for v in a) * sum(v * v for v in b))
    if not norm:
        raise ValueError("Zero vectors have no cosine direction")
    return sum(x * y for x, y in zip(a, b)) / norm


def mmr(terms, relevance, similarities, diversity=0.3, limit=3):
    n = len(terms)
    if (
        not n
        or len(set(terms)) != n
        or len(relevance) != n
        or len(similarities) != n
        or any(len(row) != n for row in similarities)
    ):
        raise ValueError(
            "Aligned unique candidates and a square similarity table are required"
        )
    if (
        not 0 <= diversity <= 1
        or type(limit) is not int
        or not 1 <= limit <= n
        or any(
            not isfinite(v)
            for v in [*relevance, *[v for row in similarities for v in row]]
        )
    ):
        raise ValueError("Invalid diversity, limit, or scores")
    selected = [max(range(n), key=lambda i: (relevance[i], -i))]
    while len(selected) < limit:
        candidates = [i for i in range(n) if i not in selected]
        selected.append(
            max(
                candidates,
                key=lambda i: (
                    (1 - diversity) * relevance[i]
                    - diversity * max(similarities[i][j] for j in selected),
                    -i,
                ),
            )
        )
    return [terms[i] for i in selected]
