"""Transparent document-level retrieval metrics, chunk offsets, and BM25."""

from collections import Counter
import math
import re


def metrics(ranking, relevance, k=3):
    """AP@k divides by all relevant documents; absent-query AP/nDCG are undefined."""
    if type(k) is not int or k < 1:
        raise ValueError("Positive integer k required")
    if len(set(ranking)) != len(ranking):
        raise ValueError("Collapse duplicate document IDs before scoring")
    if any(type(g) is not int or g < 0 for g in relevance.values()):
        raise ValueError("Nonnegative integer relevance grades required")
    total = sum(g > 0 for g in relevance.values())
    hits = 0
    terms = []
    dcg = 0.0
    rr = 0.0
    for rank, id in enumerate(ranking[:k], 1):
        grade = relevance.get(id, 0)
        if grade > 0:
            hits += 1
            terms.append(hits / rank)
            if not rr:
                rr = 1 / rank
        dcg += (2**grade - 1) / math.log2(rank + 1)
    ideal = sum(
        (2**g - 1) / math.log2(i + 2)
        for i, g in enumerate(sorted(relevance.values(), reverse=True)[:k])
    )
    return {
        "precision": hits / k,
        "recall": hits / total if total else None,
        "ap": sum(terms) / total if total else None,
        "rr": rr if total else None,
        "ndcg": dcg / ideal if ideal else None,
        "hits": hits,
        "relevant_documents": total,
        "ap_terms": terms,
        "dcg": dcg,
        "idcg": ideal,
    }


def collapse(chunks):
    seen = set()
    result = []
    for row in chunks:
        id = row["document_id"]
        if id not in seen:
            seen.add(id)
            result.append(id)
    return result


def chunks(text, size=2, overlap=1, mode="sentence"):
    if (
        type(size) is not int
        or type(overlap) is not int
        or size < 1
        or not 0 <= overlap < size
    ):
        raise ValueError("Require size>0 and 0<=overlap<size")
    if mode == "word":
        units = [m.span() for m in re.finditer(r"\S+", text)]
    elif mode == "sentence":
        units = []
        start = 0
        for match in re.finditer(r"(?<=[.!?])\s+|$", text):
            end = match.start()
            left = start
            while left < end and text[left].isspace():
                left += 1
            while end > left and text[end - 1].isspace():
                end -= 1
            if end > left:
                units.append((left, end))
            start = match.end()
    else:
        raise ValueError("Unknown chunk mode")
    result = []
    position = 0
    while position < len(units):
        end = min(position + size, len(units))
        a, b = units[position][0], units[end - 1][1]
        result.append({"start": a, "end": b, "text": text[a:b]})
        if end == len(units):
            break
        position = end - overlap
    return result


def bm25(query, documents, k1=1.2, b=0.75):
    if not documents:
        raise ValueError("Nonempty document collection required")
    tokenize = lambda text: re.findall(r"[a-z0-9]+", text.casefold())
    tokens = [tokenize(text) for text in documents]
    average = sum(map(len, tokens)) / len(tokens)
    if not average:
        raise ValueError("Collection needs tokens")
    df = Counter(t for row in tokens for t in set(row))
    scores = []
    for row in tokens:
        counts = Counter(row)
        score = 0.0
        for term in set(tokenize(query)):
            frequency = counts[term]
            idf = math.log(1 + (len(tokens) - df[term] + 0.5) / (df[term] + 0.5))
            score += (
                idf
                * frequency
                * (k1 + 1)
                / (frequency + k1 * (1 - b + b * len(row) / average))
            )
        scores.append(score)
    return scores


def citation_metrics(claims):
    """Human judgments are inputs, not inferred from citation existence."""
    links = [link for claim in claims for link in claim["citations"]]
    return {
        "citation_precision": sum(link["supports"] for link in links) / len(links)
        if links
        else None,
        "citation_recall": sum(
            any(link["supports"] for link in claim["citations"]) for claim in claims
        )
        / len(claims)
        if claims
        else None,
        "claims": len(claims),
        "citations": len(links),
    }
