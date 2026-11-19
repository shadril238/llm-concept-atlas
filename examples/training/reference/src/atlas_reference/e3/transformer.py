"""L05: explicit random-weight transformer, checked against PyTorch attention."""

import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[6]


def main():
    torch.set_num_threads(1)
    rng = np.random.default_rng(11)
    ids = np.array([0, 1, 2])
    embedding = rng.normal(size=(7, 4)) * 0.2
    x = embedding[ids]  # [T=3, d_model=4]
    norm = lambda z: (
        (z - z.mean(-1, keepdims=True)) / np.sqrt(z.var(-1, keepdims=True) + 1e-5)
    )
    normalized = norm(x)
    Wq, Wk, Wv, Wo = [rng.normal(size=(4, 4)) * 0.2 for _ in range(4)]
    split = lambda z: z.reshape(3, 2, 2).transpose(1, 0, 2)
    q, k, v = [split(normalized @ matrix) for matrix in [Wq, Wk, Wv]]
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(2)
    mask = np.tril(np.ones((3, 3), dtype=bool))
    scores = np.where(mask, scores, -np.inf)
    exponent = np.exp(scores - scores.max(-1, keepdims=True))
    weights = exponent / exponent.sum(-1, keepdims=True)
    attended = weights @ v
    reference = F.scaled_dot_product_attention(
        *[torch.tensor(z) for z in [q, k, v]], is_causal=True
    )
    error = float(np.abs(attended - reference.numpy()).max())
    assert error < 1e-10
    assert np.all(weights[:, ~mask] == 0) and np.allclose(weights.sum(-1), 1.0)
    concatenated = attended.transpose(1, 0, 2).reshape(3, 4)
    residual = x + concatenated @ Wo
    W1 = rng.normal(size=(4, 8)) * 0.2
    W2 = rng.normal(size=(8, 4)) * 0.2
    hidden = residual + np.maximum(0, norm(residual) @ W1) @ W2
    logits = hidden @ rng.normal(size=(4, 7))
    # Changing a future input cannot change earlier masked outputs.
    changed_v = v.copy()
    changed_v[:, -1, :] += 100
    assert np.allclose((weights @ changed_v)[:, :2, :], attended[:, :2, :])
    result = {
        "input_shape": list(x.shape),
        "qkv_shape": list(q.shape),
        "attention_shape": list(weights.shape),
        "head_shape": list(logits.shape),
        "weights": weights[0].round(6).tolist(),
        "values": v[0].round(6).tolist(),
        "output": hidden.round(6).tolist(),
        "logits": logits[-1].round(6).tolist(),
        "reference_max_error": error,
        "future_masked": True,
        "future_perturbation_checked": True,
        "random_weights": True,
    }
    path = ROOT / "examples/artifacts/e3-transformer"
    path.mkdir(parents=True, exist_ok=True)
    (path / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
