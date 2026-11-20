"""Evaluate externally collected hosted outputs; never makes an API request."""

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "examples/foundations/src"))
from atlas_foundations.e4.metrics import classification_metrics, parse_label


def decode_label(raw):
    if not isinstance(raw, str):
        return -1
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    if isinstance(value, dict):
        if set(value) != {"label"}:
            return -1
        value = value["label"]
    return parse_label(value)


def evaluate_external(fixture, collected):
    for field in ["provider", "model", "prompt_version", "collected_at"]:
        if not isinstance(collected.get(field), str) or not collected[field].strip():
            raise ValueError(f"Missing provenance: {field}")
    test = [row for row in fixture["rows"] if row["split"] == "test"]
    expected = {r["id"] for r in test}
    found = {}
    for row in collected.get("outputs", []):
        if row.get("id") not in expected or row["id"] in found:
            raise ValueError("Unknown or duplicate test ID")
        if row.get("latency_seconds") is not None and (
            not isinstance(row["latency_seconds"], (float, int))
            or not math.isfinite(row["latency_seconds"])
            or row["latency_seconds"] < 0
        ):
            raise ValueError("Latency must be a finite nonnegative measurement")
        found[row["id"]] = row
    predicted = [
        decode_label(found[row["id"]].get("raw"))
        if row["id"] in found and not found[row["id"]].get("error")
        else -1
        for row in test
    ]
    return {
        "status": "externally-collected; provider execution not verified by atlas",
        "provenance": {
            key: collected[key]
            for key in ["provider", "model", "prompt_version", "collected_at"]
        },
        "metrics": classification_metrics([r["label"] for r in test], predicted),
        "missing_outputs": len(expected - found.keys()),
        "provider_errors": sum(bool(row.get("error")) for row in found.values()),
        "cases": [dict(id=row["id"], prediction=p) for row, p in zip(test, predicted)],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    fixture = json.loads((ROOT / "examples/data/fixtures/sentiment.json").read_text())
    result = evaluate_external(fixture, json.loads(args.results.read_text()))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
