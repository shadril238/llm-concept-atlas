"""Small pretrained embedding smoke run; not a retrieval benchmark."""
# region demonstration
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "sentence-transformers/paraphrase-MiniLM-L3-v2"
REVISION = "4ca70771034acceecb2e72475f72050fcdde4ddc"
ROOT = Path(__file__).resolve().parents[4]


def main() -> None:
    torch.manual_seed(0)
    torch.set_num_threads(1)
    # A local snapshot can be supplied for a fully offline rerun.
    location = os.environ.get("ATLAS_MODEL_PATH", MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(location, revision=REVISION)
    model = AutoModel.from_pretrained(location, revision=REVISION).eval().to("cpu")
    data = json.loads((ROOT / "examples/data/fixtures/retrieval.json").read_text())
    texts = [row["text"] for row in data["documents"] + data["queries"]]
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.inference_mode():
        hidden = model(**encoded).last_hidden_state  # [5, sequence length, 384]
        mask = encoded["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        vectors = torch.nn.functional.normalize(pooled, p=2, dim=1).numpy()
    assert vectors.shape == (5, 384) and np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    scores = vectors[3:] @ vectors[:3].T
    rankings = [data["documents"][int(index)]["id"] for index in scores.argmax(axis=1)]
    expected = [query["relevant"] for query in data["queries"]]
    assert rankings == expected, (rankings, expected)
    artifact = Path(os.environ.get("ATLAS_ARTIFACT_DIR", str(ROOT / "examples/artifacts/retrieval")))
    artifact.mkdir(parents=True, exist_ok=True)
    np.save(artifact / "embeddings.npy", vectors)
    assert np.allclose(np.load(artifact / "embeddings.npy"), vectors)
    print(json.dumps({"shape": list(vectors.shape), "top_documents": rankings,
                      "cosine_scores": [[round(float(value), 4) for value in row] for row in scores],
                      "fixture_hits": len(rankings), "fixture_queries": len(expected)}))


if __name__ == "__main__":
    main()
# endregion demonstration
