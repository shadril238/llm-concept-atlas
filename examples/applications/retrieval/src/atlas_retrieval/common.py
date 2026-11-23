"""Shared pinned CPU model adapters and source-preserving retrieval operations."""

from pathlib import Path
import json
import os
import sys
import time
import functools

ROOT = Path(__file__).resolve().parents[5]
PROJECT = ROOT / "examples/applications/retrieval"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
sys.path.insert(0, str(ROOT / "examples/foundations/src"))
sys.path.insert(0, str(PROJECT))
from download_models import download
from atlas_foundations.e6.retrieval import chunks, bm25, collapse, metrics
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
)
from sklearn.neighbors import NearestNeighbors

torch.set_num_threads(1)
torch.manual_seed(42)
MODELS = json.loads((PROJECT / "models.json").read_text())


def fixture():
    return json.loads(
        (ROOT / "examples/data/fixtures/retrieval-corpus.json").read_text()
    )


def save(name: str, result: dict):
    directory = ROOT / "examples/artifacts" / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


@functools.lru_cache(maxsize=None)
def snapshot(name: str):
    return download(name)


def provenance(names):
    return {
        name: {
            "revision": MODELS[name]["revision"],
            "artifact_format": "safetensors",
            "remote_code": False,
        }
        for name in names
    }


class Encoder:
    name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self):
        path = snapshot(self.name)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModel.from_pretrained(
            path, local_files_only=True, use_safetensors=True
        ).eval()
        self.cap = 96

    def encode(self, texts: list[str]) -> np.ndarray:
        counts = [len(self.tokenizer.encode(text)) for text in texts]
        if max(counts, default=0) > self.cap:
            raise ValueError(
                "A retrieval input exceeds the declared cap; do not silently truncate"
            )
        vectors = []
        for offset in range(0, len(texts), 8):
            batch = self.tokenizer(
                texts[offset : offset + 8],
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            with torch.inference_mode():
                hidden = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                vectors.append(torch.nn.functional.normalize(pooled, p=2, dim=-1).cpu())
        result = torch.cat(vectors).numpy()
        assert np.isfinite(result).all() and np.allclose(
            np.linalg.norm(result, axis=1), 1, atol=1e-5
        )
        return result


class Reranker:
    name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self):
        path = snapshot(self.name)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path, local_files_only=True, use_safetensors=True
        ).eval()

    def score(self, query: str, texts: list[str]) -> list[float]:
        batch = self.tokenizer(
            [query] * len(texts),
            texts,
            padding=True,
            truncation=True,
            max_length=192,
            return_tensors="pt",
        )
        with torch.inference_mode():
            scores = self.model(**batch).logits[:, 0]
        assert torch.isfinite(scores).all()
        return scores.tolist()  # raw logits; not calibrated relevance probabilities


class Generator:
    name = "google/flan-t5-small"

    def __init__(self):
        path = snapshot(self.name)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            path, local_files_only=True, use_safetensors=True
        ).eval()

    def generate(self, prompt: str, max_tokens: int = 48) -> dict:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=False)
        if inputs["input_ids"].shape[1] > 512:
            raise ValueError("Generation input exceeds the declared 512-token cap")
        start = time.perf_counter()
        with torch.inference_mode():
            ids = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=max_tokens
            )
        return {
            "prompt": prompt,
            "raw": self.tokenizer.decode(ids[0], skip_special_tokens=True),
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "output_tokens": int(ids.shape[1] - 1),
            "seconds": time.perf_counter() - start,
            "settings": {
                "do_sample": False,
                "max_new_tokens": max_tokens,
                "input_cap": 512,
            },
            "output_ids": ids[0].tolist(),
        }

    def choose(self, prompt: str, choices: list[str]) -> dict:
        """Score a fixed shared label domain; this is not free-form constrained decoding."""
        source = self.tokenizer(
            [prompt] * len(choices), padding=True, return_tensors="pt", truncation=False
        )
        if source["input_ids"].shape[1] > 512:
            raise ValueError("Choice input cap exceeded")
        target = self.tokenizer(choices, padding=True, return_tensors="pt")["input_ids"]
        labels = target.masked_fill(target == self.tokenizer.pad_token_id, -100)
        started = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(**source, labels=labels).logits
            loss = torch.nn.functional.cross_entropy(
                logits.transpose(1, 2), labels, reduction="none", ignore_index=-100
            )
            count = (labels != -100).sum(1)
            scores = -loss.sum(1) / count
        best = int(scores.argmax())
        return {
            "kind": "finite-label likelihood scoring",
            "prompt": prompt,
            "raw": choices[best],
            "choices": choices,
            "mean_log_probabilities": scores.tolist(),
            "prompt_tokens": int(source["attention_mask"].sum()),
            "unique_prompt_tokens": int(source["attention_mask"][0].sum()),
            "output_tokens": 0,
            "scored_tokens": int(count.sum()),
            "candidate_batch_size": len(choices),
            "seconds": time.perf_counter() - started,
            "settings": {
                "generated_tokens": False,
                "length_normalization": "mean target-token log probability including EOS",
                "domain": "predeclared collection names; finish permitted after evidence only",
            },
        }


def make_chunks(documents, policy="sentence"):
    result = []
    for document in documents:
        for i, chunk in enumerate(
            chunks(
                document["text"],
                2 if policy == "sentence" else 24,
                1 if policy == "sentence" else 6,
                policy,
            )
        ):
            result.append(
                {
                    **chunk,
                    "id": f"{document['id']}-c{i}",
                    "document_id": document["id"],
                    "title": document["title"],
                    "collection": document["collection"],
                    "source_uri": document["source_uri"],
                }
            )
    return result


class Retriever:
    def __init__(self, encoder: Encoder, documents, policy="sentence"):
        self.encoder = encoder
        self.chunks = make_chunks(documents, policy)
        self.texts = [r["title"] + ": " + r["text"] for r in self.chunks]
        self.vectors = encoder.encode(self.texts)
        self.index = NearestNeighbors(metric="cosine", algorithm="brute").fit(
            self.vectors
        )

    def search(
        self, query: str, method="dense", limit=6, collection=None, reranker=None
    ):
        if method == "lexical":
            scores = bm25(query, self.texts)
            ids = sorted(
                range(len(scores)), key=lambda i: (-scores[i], self.chunks[i]["id"])
            )
            if collection:
                ids = [i for i in ids if self.chunks[i]["collection"] == collection]
            return [
                {**self.chunks[i], "score": scores[i], "dense_score": None}
                for i in ids[:limit]
            ]
        vector = self.encoder.encode([query])
        distances, indices = self.index.kneighbors(vector, n_neighbors=len(self.chunks))
        # Independent exact dot products check score direction and native index contract.
        independent = [
            float(torch.dot(torch.from_numpy(vector[0]), torch.from_numpy(row)))
            for row in self.vectors
        ]
        assert all(
            abs((1 - float(distance)) - independent[int(i)]) < 2e-6
            for distance, i in zip(distances[0], indices[0])
        )
        if method == "lexical":
            scores = bm25(query, self.texts)
        else:
            scores = independent
        ids = sorted(
            range(len(scores)), key=lambda i: (-scores[i], self.chunks[i]["id"])
        )
        if collection:
            ids = [i for i in ids if self.chunks[i]["collection"] == collection]
        if method == "reranked":
            ids = ids[:6]
            pair_scores = reranker.score(query, [self.texts[i] for i in ids])
            score_by_id = dict(zip(ids, pair_scores))
            ids.sort(key=lambda i: (-score_by_id[i], self.chunks[i]["id"]))
            return [
                {
                    **self.chunks[i],
                    "score": score_by_id[i],
                    "dense_score": independent[i],
                }
                for i in ids[:limit]
            ]
        return [
            {**self.chunks[i], "score": scores[i], "dense_score": independent[i]}
            for i in ids[:limit]
        ]
