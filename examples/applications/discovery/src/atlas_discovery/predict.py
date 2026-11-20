"""Use the saved L07 JSON head; no estimator pickle or retraining required."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas_discovery.common import ROOT, EMBEDDER, EMBED_REV, Encoder, setup
import numpy as np


def main(argv=None):
    setup()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text")
    args = parser.parse_args(argv)
    path = ROOT / "examples/artifacts/e4-classification/classifier.json"
    if not path.exists():
        raise ValueError("Run classification-comparison first")
    saved = json.loads(path.read_text())
    assert (
        saved["embedding_model"] == EMBEDDER
        and saved["embedding_revision"] == EMBED_REV
    )
    assert saved["classes"] == [0, 1] and len(saved["coef"][0]) == 384
    vector = Encoder().encode([args.text])
    z = float((vector @ np.asarray(saved["coef"]).T + saved["intercept"])[0, 0])
    positive = float(1 / (1 + np.exp(-z)))
    predicted = int(positive >= 0.5)
    confidence = max(positive, 1 - positive)
    print(
        json.dumps(
            {
                "label": ["negative", "positive"][predicted],
                "score": confidence,
                "human_review": saved.get("always_review", False)
                or confidence < saved["review_threshold"],
                "review_threshold": saved["review_threshold"],
                "warning": "This score is not a guarantee of correctness or domain validity.",
            }
        )
    )


if __name__ == "__main__":
    main()
