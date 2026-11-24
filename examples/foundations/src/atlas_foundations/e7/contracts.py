"""Inspectable training-data, loss, corruption and alignment contracts; no model dependency."""

import math
import random
from collections import Counter


def cross_entropy(logits, target):
    if (
        type(target) is not int
        or not 0 <= target < len(logits)
        or not logits
        or not all(math.isfinite(x) for x in logits)
    ):
        raise ValueError("Finite logits and an integer target index required")
    top = max(logits)
    return top + math.log(sum(math.exp(x - top) for x in logits)) - logits[target]


def mnr_loss(scores, scale=20):
    if (
        not math.isfinite(scale)
        or scale <= 0
        or len(scores) < 2
        or any(len(row) != len(scores) for row in scores)
    ):
        raise ValueError("Positive scale and square paired score matrix required")
    terms = [cross_entropy([scale * x for x in row], i) for i, row in enumerate(scores)]
    return {"terms": terms, "loss": sum(terms) / len(terms)}


def check_splits(data):
    rows = data["rows"]
    by_id = {r["id"]: r for r in rows}
    if len(by_id) != len(rows) or len(
        {r["text"].strip().casefold() for r in rows}
    ) != len(rows):
        raise ValueError("Duplicate IDs or text across splits")
    for p in data["gold_pairs"]:
        if p["split"] not in ["train", "validation", "test"] or any(
            by_id[p[k]]["split"] != p["split"] for k in ["left_id", "right_id"]
        ):
            raise ValueError("Gold pair crosses splits")
    for p in data["silver_candidates"]:
        if any(by_id[p[k]]["split"] != "silver-pool" for k in ["left_id", "right_id"]):
            raise ValueError("Silver candidate contains evaluation text")
    return True


def check_mnr_batch(pairs, by_id):
    intents = []
    texts = []
    for p in pairs:
        a, b = by_id[p["left_id"]], by_id[p["right_id"]]
        if a["intent"] != b["intent"] or a["split"] != "train" or b["split"] != "train":
            raise ValueError("Expected a positive training pair")
        intents.append(a["intent"])
        texts.extend([a["text"], b["text"]])
    if len(set(intents)) != len(intents) or len(set(texts)) != len(texts):
        raise ValueError("False negative or duplicate in MNR batch")
    return True


def delete_words(text, seed, probability=0.6):
    if not 0 <= probability < 1:
        raise ValueError("Deletion probability must be in [0,1)")
    words = text.split()
    if not words:
        raise ValueError("Nonempty sentence required")
    rng = random.Random(seed)
    indices = [i for i in range(len(words)) if rng.random() >= probability]
    if not indices:
        indices = [rng.randrange(len(words))]
    return {
        "original": text,
        "damaged": " ".join(words[i] for i in indices),
        "retained_word_indices": indices,
        "deletion_probability": probability,
        "seed": seed,
    }


def align_all(word_ids, tags, counts):
    from atlas_foundations.ner_contract import LABELS

    retained = Counter(i for i in word_ids if i is not None)
    if retained != counts:
        raise ValueError("Truncated record: do not train a shortened entity")
    seen = set()
    labels = []
    for i in word_ids:
        if i is None:
            labels.append(-100)
            continue
        tag = tags[i]
        if i in seen and tag.startswith("B-"):
            tag = "I-" + tag[2:]
        labels.append(LABELS.index(tag))
        seen.add(i)
    return labels
