"""L03: pretrained static words, contextual tokens, and pooled sentences."""

import gzip
import hashlib
import json
import os
from pathlib import Path
import urllib.request
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[6]
GLOVE_URL = "https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz"
GLOVE_SHA = "5c55f98957aa9fed8d2ac5fb1dcff57af3b23c5a3ee7af3f7945f8d49198eb24"
BERT = "prajjwal1/bert-tiny"
BERT_REV = "6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837"
TOK = "google-bert/bert-base-uncased"
TOK_REV = "86b5e0934494bd15c9632b12f734a8a67f723594"
SENTENCE = "sentence-transformers/paraphrase-MiniLM-L3-v2"
SENTENCE_REV = "4ca70771034acceecb2e72475f72050fcdde4ddc"


def main():
    torch.set_num_threads(1)
    path = ROOT / ".cache/e3/glove-wiki-gigaword-50.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        if os.getenv("HF_HUB_OFFLINE") == "1":
            raise RuntimeError(
                "GloVe cache missing: rerun explicitly with --allow-network"
            )
        urllib.request.urlretrieve(GLOVE_URL, path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == GLOVE_SHA
    words = []
    with gzip.open(path, "rt", encoding="utf8") as stream:
        rows, width = map(int, next(stream).split())
        vectors = np.empty((rows, width), dtype=np.float32)
        for i, line in enumerate(stream):
            fields = line.split()
            words.append(fields[0])
            vectors[i] = np.asarray(fields[1:], dtype=np.float32)
    index = {word: i for i, word in enumerate(words)}
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    neighbors = {}
    for query in ["king", "bank", "good"]:
        scores = vectors @ vectors[index[query]]
        scores[index[query]] = -np.inf
        neighbors[query] = [
            {"word": words[i], "cosine": round(float(scores[i]), 5)}
            for i in np.argsort(-scores)[:5]
        ]
    tokenizer = AutoTokenizer.from_pretrained(TOK, revision=TOK_REV, use_fast=True)
    model = AutoModel.from_pretrained(BERT, revision=BERT_REV).eval()
    sentences = ["The bank approved the loan.", "The bank flooded after rain."]
    encoded = tokenizer(sentences, padding=True, return_tensors="pt")
    with torch.inference_mode():
        lookup = model.get_input_embeddings()(encoded.input_ids)
        hidden = model(**encoded).last_hidden_state
    positions = [
        ids.index(tokenizer.convert_tokens_to_ids("bank"))
        for ids in encoded.input_ids.tolist()
    ]
    a, b = [hidden[i, pos] for i, pos in enumerate(positions)]
    assert torch.equal(lookup[0, positions[0]], lookup[1, positions[1]])
    assert not torch.allclose(a, b)
    contextual_cosine = float(
        torch.nn.functional.cosine_similarity(a[None], b[None])[0]
    )
    st = AutoTokenizer.from_pretrained(SENTENCE, revision=SENTENCE_REV)
    sm = AutoModel.from_pretrained(SENTENCE, revision=SENTENCE_REV).eval()
    texts = sentences + ["A lender accepted a credit application."]
    batch = st(texts, padding=True, return_tensors="pt")
    with torch.inference_mode():
        states = sm(**batch).last_hidden_state
        mask = batch.attention_mask.unsqueeze(-1)
        pooled = (states * mask).sum(1) / mask.sum(1)
        pooled = torch.nn.functional.normalize(pooled, dim=1)
    similarities = (pooled @ pooled.T).tolist()
    selected = ["king", "queen", "bank", "river", "loan", "good", "bad", "excellent"]
    small = vectors[[index[w] for w in selected]]
    centered = small - small.mean(0)
    _, singular, rotation = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ rotation[:2].T
    result = {
        "static_shape": [rows, width],
        "neighbors": neighbors,
        "bank_token_id": tokenizer.convert_tokens_to_ids("bank"),
        "lookup_shape": list(lookup.shape),
        "hidden_shape": list(hidden.shape),
        "same_lookup": True,
        "different_context": True,
        "bank_context_cosine": contextual_cosine,
        "sentence_shape": list(pooled.shape),
        "sentence_cosines": similarities,
        "sentences": texts,
        "projection": {
            "words": selected,
            "coordinates": projection.tolist(),
            "variance_fraction": float((singular[:2] ** 2).sum() / (singular**2).sum()),
            "method": "PCA of eight selected normalized 50D GloVe rows; nearest neighbors computed in 50D, not this projection",
        },
        "numerical_token_counts": {
            name: len(tok("12.0*50=600\n\t42!", add_special_tokens=True).input_ids)
            for name, tok in [("bert", tokenizer), ("sentence", st)]
        },
        "glove_sha256": GLOVE_SHA,
    }
    out = ROOT / "examples/artifacts/e3-representations"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
