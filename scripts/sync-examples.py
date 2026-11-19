#!/usr/bin/env python3
"""Generate lesson/notebook views from named Python regions and recorded outputs."""

import argparse
import json
from pathlib import Path
from example_tools import ROOT, load, region, render_lab, validate

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--check", action="store_true")
args = parser.parse_args()
manifest = load()
errors = validate(manifest, check_generated=args.check)
if errors:
    raise SystemExit("\n".join(errors))
outputs = {"src/content/docs/labs/first-verified-examples.mdx": render_lab(manifest)}
report = [
    "# Example verification report",
    "",
    "Generated from the registry and execution records. Execution checks do not constitute lesson acceptance.",
    "",
    "| Example | Level | Validation | Execution evidence |",
    "| --- | --- | --- | --- |",
]
for entry in manifest["examples"]:
    evidence = "Unverified"
    if entry["verificationRecord"]:
        record = json.loads((ROOT / entry["verificationRecord"]).read_text())
        evidence = f"[Passed {record['date']}]({'records/' + Path(entry['verificationRecord']).name})"
    report.append(
        f"| {entry['id']} | {entry['level']} | {entry['validation']['method']} | {evidence} |"
    )
outputs["examples/verification/README.md"] = "\n".join(report) + "\n"
for family, eid in [
    ("foundations", "attention-softmax"),
    ("applications", "embedding-retrieval"),
    ("reference", "tokenizer-comparison"),
    ("reference", "lora-tiny-training"),
    ("reference", "lora-pretrained"),
    ("reference", "ner-training"),
]:
    entry = next(e for e in manifest["examples"] if e["id"] == eid)
    directory = ROOT / manifest["environments"][family]["project"]
    module = {
        "attention-softmax": "atlas_foundations.attention",
        "embedding-retrieval": "atlas_applications.retrieval",
        "tokenizer-comparison": "atlas_reference.tokenizer_compare",
        "lora-tiny-training": "atlas_reference.lora_tiny",
        "lora-pretrained": "atlas_reference.lora_model",
        "ner-training": "atlas_reference.ner_model",
    }[eid]
    # Generated notebooks import the same source used by the CLI; no copied algorithm.
    code = f'import sys\nfrom pathlib import Path\nproject = Path.cwd()\nif project.name == "notebooks": project = project.parent\nsys.path.insert(0, str(project / "src"))\nfrom {module} import main\nmain()'
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": f"Python 3.11 ({family})",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "cells": [
            {
                "cell_type": "markdown",
                "id": "introduction",
                "metadata": {},
                "source": [
                    f"# {entry['title']}\n",
                    "Run from this environment directory or its notebooks subdirectory. Install the locked environment first.\n",
                    "Predict the result before running. Read the generated lesson for interpretation, limits, and transfer questions.",
                ],
            },
            {
                "cell_type": "code",
                "id": "run-source",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": code.splitlines(keepends=True),
            },
        ],
    }
    outputs[f"{manifest['environments'][family]['project']}/notebooks/{eid}.ipynb"] = (
        json.dumps(notebook, indent=2) + "\n"
    )
if not args.check:
    artifacts = {"tokenizers": "tokenizers", "lora": "lora-model", "ner": "ner-model"}
    paths = {
        key: ROOT / "examples/artifacts" / directory / "results.json"
        for key, directory in artifacts.items()
    }
    if all(path.exists() for path in paths.values()):
        results = {key: json.loads(path.read_text()) for key, path in paths.items()}
        results["tiny"] = json.loads(
            json.loads(
                (
                    ROOT / "examples/verification/records/lora-tiny-training.json"
                ).read_text()
            )["stdout"]
        )
        outputs["src/data/reference-results.json"] = (
            json.dumps(results, indent=2, ensure_ascii=False) + "\n"
        )
# Replace only explicit source regions; surrounding explanations stay authored.
for entry in manifest["examples"]:
    if not entry.get("lesson"):
        continue
    name = entry["lesson"]["path"]
    path = ROOT / name
    text = outputs.get(name, path.read_text())
    start, end = (
        f"{{/* example:{entry['id']}:start */}}",
        f"{{/* example:{entry['id']}:end */}}",
    )
    if start in text:
        before, remainder = text.split(start, 1)
        _, after = remainder.split(end, 1)
        outputs[name] = (
            before
            + start
            + "\n\n```python\n"
            + region(entry)
            + "\n```\n\n"
            + end
            + after
        )
for name, content in outputs.items():
    path = ROOT / name
    if args.check:
        if not path.exists() or path.read_text() != content:
            raise SystemExit(
                f"Generated example view is stale: {name}; run npm run examples:sync"
            )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
print(
    f"Validated {len(manifest['examples'])} examples and {len(outputs)} synchronized views."
)
