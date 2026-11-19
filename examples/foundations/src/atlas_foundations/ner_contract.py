"""Word-level BIO alignment and strict, sentence-preserving entity scoring."""

LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


def spans(tags):
    """Return (type, start, exclusive_end). Invalid I starts a new span.

    This explicit repair policy is applied to both displayed predictions and
    metrics. Never join spans across sentence boundaries.
    """
    result, active, start = [], None, 0
    for index, tag in enumerate(list(tags) + ["O"]):
        prefix, kind = ("O", None) if tag == "O" else tag.split("-", 1)
        if prefix not in ["O", "B", "I"]:
            raise ValueError("Expected BIO tags")
        continues = prefix == "I" and kind == active
        if active is not None and not continues:
            result.append((active, start, index))
            active = None
        if prefix != "O" and not continues:
            active, start = kind, index
    return result


def entity_scores(gold, predicted):
    if len(gold) != len(predicted) or any(
        len(a) != len(b) for a, b in zip(gold, predicted)
    ):
        raise ValueError("Sentence and word counts must match")
    expected = {(i, *span) for i, tags in enumerate(gold) for span in spans(tags)}
    actual = {(i, *span) for i, tags in enumerate(predicted) for span in spans(tags)}
    tp = len(expected & actual)
    precision = tp / len(actual) if actual else 0.0
    recall = tp / len(expected) if expected else 0.0
    return {
        "tp": tp,
        "predicted": len(actual),
        "gold": len(expected),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
    }


def align_first(word_ids, tags, complete_counts, reject_truncated=True):
    """Only a retained complete word's first piece receives loss supervision.

    Reject rows with any truncated word/entity: no partial entity can silently
    become a shorter gold entity. Interactive inspection can show masks without
    training on such a row.
    """
    from collections import Counter

    retained = Counter(index for index in word_ids if index is not None)
    if reject_truncated and retained != complete_counts:
        raise ValueError("Truncated record: chunk at safe entity boundaries first")
    seen, labels = set(), []
    for index in word_ids:
        if index is None or index in seen or retained[index] != complete_counts[index]:
            labels.append(-100)
        else:
            labels.append(LABELS.index(tags[index]))
            seen.add(index)
    return labels


# region demonstration
def main():
    gold = [["B-PER", "O", "B-ORG", "I-ORG", "O"]]
    boundary_error = [["B-PER", "O", "B-ORG", "O", "O"]]
    result = entity_scores(gold, boundary_error)
    print(
        "token accuracy:", sum(a == b for a, b in zip(gold[0], boundary_error[0])) / 5
    )
    print("strict entity F1:", result["f1"])
    assert result["tp"] == 1 and result["gold"] == 2 and result["f1"] == 0.5


if __name__ == "__main__":
    main()
# endregion demonstration
