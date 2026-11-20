"""Pinned CPU artifacts shared by the E4 workflows."""

import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "examples/foundations/src"))
EMBEDDER = "sentence-transformers/paraphrase-MiniLM-L3-v2"
EMBED_REV = "4ca70771034acceecb2e72475f72050fcdde4ddc"
SENTIMENT = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
SENTIMENT_REV = "714eb0fa89d2f80546fda750413ed43d93601a13"
NLI = "cross-encoder/nli-MiniLM2-L6-H768"
NLI_REV = "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d"
T5 = "google/flan-t5-small"
T5_REV = "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"


def setup():
    global _thread_limit
    _thread_limit = threadpool_limits(limits=1)
    torch.manual_seed(37)
    np.random.seed(37)
    torch.set_num_threads(1)


def save_result(name, result):
    path = ROOT / "examples/artifacts" / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


class Encoder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDER, revision=EMBED_REV)
        self.model = AutoModel.from_pretrained(EMBEDDER, revision=EMBED_REV).eval()

    def encode(self, texts):
        outputs = []
        for start in range(0, len(texts), 16):
            batch = self.tokenizer(
                list(texts[start : start + 16]),
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            with torch.inference_mode():
                states = self.model(**batch).last_hidden_state
                mask = batch.attention_mask.unsqueeze(-1)
                pooled = (states * mask).sum(1) / mask.sum(1)
                outputs.append(torch.nn.functional.normalize(pooled, dim=1).numpy())
        vectors = np.concatenate(outputs)
        assert np.isfinite(vectors).all()
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
        return vectors


def measure(predict, rows):
    # CPU eager work finishes synchronously. Loading and warm-up are excluded.
    predict(rows[:2])
    timings = []
    for _ in range(3):
        start = time.perf_counter()
        result = []
        for offset in range(0, len(rows), 16):
            result.extend(predict(rows[offset : offset + 16]))
        timings.append(time.perf_counter() - start)
    return result, {
        "seconds": timings,
        "median_seconds": float(np.median(timings)),
        "rows": len(rows),
        "batch_size": 16,
        "device": "cpu",
        "threads": 1,
        "includes_loading": False,
        "includes_tokenization_or_feature_extraction": True,
    }
