"""Manifest validation and source synchronization shared by CLI and tests."""

import hashlib
import json
import re
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load():
    return json.loads((ROOT / "examples/manifest.json").read_text())


def code_blocks(text):
    return re.findall(r"```python\n([\s\S]*?)\n```", text)


def region(entry, root=ROOT):
    text = (root / entry["entrypoint"]).read_text()
    if entry["region"]:
        name = re.escape(entry["region"])
        matches = re.findall(
            rf"^[ \t]*# region {name}\n([\s\S]*?)^[ \t]*# endregion {name}$", text, re.M
        )
        if len(matches) != 1:
            raise ValueError(f"{entry['id']}: region must occur exactly once")
        return textwrap.dedent(matches[0]).rstrip()
    return text.rstrip()


def schema_errors(value, schema, label="$"):
    """Validate the JSON Schema keywords used by our versioned contracts."""
    errors = []
    types = {
        "object": dict,
        "array": list,
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "null": type(None),
    }
    expected = schema.get("type")
    if expected:
        choices = expected if isinstance(expected, list) else [expected]
        if not any(
            isinstance(value, types[k])
            and not (k in ["integer", "number"] and isinstance(value, bool))
            for k in choices
        ):
            return [f"{label}: invalid schema type"]
    if "const" in schema and value != schema["const"]:
        errors.append(f"{label}: invalid constant")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{label}: invalid enum")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{label}: missing {key}")
        props = schema.get("properties", {})
        for key, item in value.items():
            if key in props:
                errors.extend(schema_errors(item, props[key], label + "." + key))
            elif schema.get("additionalProperties") is False:
                errors.append(f"{label}: unknown field {key}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{label}: too few items")
        for i, item in enumerate(value):
            errors.extend(schema_errors(item, schema.get("items", {}), f"{label}[{i}]"))
    if (
        isinstance(value, str)
        and "pattern" in schema
        and not re.search(schema["pattern"], value)
    ):
        errors.append(f"{label}: invalid pattern")
    if isinstance(value, (int, float)):
        if value < schema.get("minimum", float("-inf")) or value > schema.get(
            "maximum", float("inf")
        ):
            errors.append(f"{label}: outside schema bounds")
    return errors


def validate(manifest, root=ROOT, check_records=True, check_generated=True):
    errors = schema_errors(
        manifest,
        json.loads((root / "examples/schemas/manifest.schema.json").read_text()),
    )
    if errors:
        return errors
    ids, registered = set(), set()
    source_ids = {
        s["id"]
        for s in json.loads((root / "src/data/book-sections.json").read_text())[
            "sections"
        ]
    }
    concepts = {p.stem for p in (root / "src/content/docs/concepts").glob("*.mdx")}
    required = {
        "id",
        "title",
        "concepts",
        "sourceSections",
        "level",
        "kind",
        "entrypoint",
        "environment",
        "workingDirectory",
        "lesson",
        "region",
        "models",
        "dataset",
        "seed",
        "reproducibility",
        "runtime",
        "network",
        "auth",
        "inputSchema",
        "outputSchema",
        "validation",
        "verificationRecord",
        "troubleshooting",
    }

    def safe(value):
        return (
            isinstance(value, str)
            and not Path(value).is_absolute()
            and ".." not in Path(value).parts
        )

    for entry in manifest["examples"]:
        eid = entry.get("id", "?")
        if required - entry.keys():
            errors.append(f"{eid}: missing fields")
            continue
        if eid in ids:
            errors.append(f"{eid}: duplicate ID")
        ids.add(eid)
        if not re.fullmatch(r"[a-z0-9-]+", eid):
            errors.append(f"{eid}: invalid ID")
        if entry["level"] not in ["A", "B", "C"]:
            errors.append(f"{eid}: invalid level")
        if entry["environment"] not in manifest["environments"]:
            errors.append(f"{eid}: unknown environment")
        if not safe(entry["entrypoint"]) or not (root / entry["entrypoint"]).is_file():
            errors.append(f"{eid}: missing or unsafe entrypoint")
            continue
        if (
            not safe(entry["workingDirectory"])
            or not (root / entry["workingDirectory"]).is_dir()
        ):
            errors.append(f"{eid}: unsafe working directory")
        if not set(entry["concepts"]) <= concepts:
            errors.append(f"{eid}: unknown concept")
        if (
            not entry["sourceSections"]
            or not set(entry["sourceSections"]) <= source_ids
        ):
            errors.append(f"{eid}: missing/unknown source sections")
        if not 0 < entry["runtime"]["timeoutSeconds"] <= 3600:
            errors.append(f"{eid}: invalid timeout")
        if entry["validation"]["method"] not in [
            "exact-stdout",
            "exit-success",
            "json-invariants",
        ]:
            errors.append(f"{eid}: unknown validation")
        if entry["network"] not in ["offline", "downloads", "first-download"]:
            errors.append(f"{eid}: unknown network policy")
        for model in entry["models"]:
            if not re.fullmatch("[a-f0-9]{40}", model["revision"]):
                errors.append(f"{eid}: unpinned model")
        if (
            entry["environment"] != "legacy-unverified"
            and entry["level"] != "A"
            and not entry["models"]
        ):
            errors.append(f"{eid}: model workflow lacks pinned model")
        if entry["dataset"]:
            if not safe(entry["dataset"]):
                errors.append(f"{eid}: unsafe fixture manifest")
                continue
            fixture = json.loads((root / entry["dataset"]).read_text())
            errors.extend(
                schema_errors(
                    fixture,
                    json.loads(
                        (root / "examples/schemas/fixture.schema.json").read_text()
                    ),
                )
            )
            if (
                not safe(fixture["path"])
                or digest(root / fixture["path"]) != fixture["sha256"]
            ):
                errors.append(f"{eid}: fixture hash mismatch")
        try:
            source = region(entry, root)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if entry["lesson"] and (check_generated or entry["id"].startswith("legacy-")):
            lesson = entry["lesson"]
            key = (lesson["path"], lesson["block"])
            if key in registered:
                errors.append(f"{eid}: duplicate lesson block")
            registered.add(key)
            if not safe(lesson["path"]):
                errors.append(f"{eid}: unsafe lesson path")
                continue
            lesson_text = (root / lesson["path"]).read_text()
            blocks = code_blocks(lesson_text)
            if (
                lesson["block"] >= len(blocks)
                or source != blocks[lesson["block"]].rstrip()
            ):
                errors.append(f"{eid}: lesson/source drift")
        for helper in entry.get("supportFiles", []):
            if not safe(helper) or not (root / helper).is_file():
                errors.append(f"{eid}: unsafe or missing support source")
        if entry["verificationRecord"] and check_records:
            path = entry["verificationRecord"]
            if not safe(path) or not (root / path).is_file():
                errors.append(f"{eid}: missing record")
                continue
            record = json.loads((root / path).read_text())
            record_errors = schema_errors(
                record,
                json.loads(
                    (root / "examples/schemas/execution-record.schema.json").read_text()
                ),
            )
            errors.extend(record_errors)
            if record_errors:
                continue
            if record["exampleId"] != eid or record["status"] != "passed":
                errors.append(f"{eid}: invalid verification record")
            if record.get("supportHashes", {}) != {
                name: digest(root / name) for name in entry.get("supportFiles", [])
            }:
                errors.append(f"{eid}: stale verification support source")
            if check_generated and eid == "lora-tiny-training":
                published = json.loads((root / "src/data/reference-results.json").read_text())["tiny"]
                if published != json.loads(record["stdout"]):
                    errors.append(f"{eid}: published regression result differs from verified output")
            if check_generated and entry.get("resultKey"):
                published = json.loads(
                    (root / "src/data/reference-results.json").read_text()
                )[entry["resultKey"]]
                encoded = json.dumps(published, indent=2, ensure_ascii=False) + "\n"
                if hashlib.sha256(encoded.encode()).hexdigest() != record.get(
                    "artifactSha256"
                ):
                    errors.append(
                        f"{eid}: published result differs from verified artifact"
                    )
            if record["sourceSha256"] != digest(root / entry["entrypoint"]):
                errors.append(f"{eid}: stale verification source")
            project = manifest["environments"][entry["environment"]]["project"]
            if record["lockSha256"] != digest(root / project / "uv.lock"):
                errors.append(f"{eid}: stale verification environment")
            if (
                entry["dataset"]
                and record["fixtureSha256"]
                != json.loads((root / entry["dataset"]).read_text())["sha256"]
            ):
                errors.append(f"{eid}: stale verification fixture")
            if record["contractSha256"] != contract_digest(entry):
                errors.append(f"{eid}: stale verification contract")
    for p in (root / "src/content/docs").rglob("*.mdx"):
        for i, _ in enumerate(code_blocks(p.read_text())):
            if check_generated and (str(p.relative_to(root)), i) not in registered:
                errors.append(f"{p.name}: unregistered Python block {i}")
    for name, env in manifest["environments"].items():
        if name != "legacy-unverified" and (
            not (root / env["lock"]).is_file()
            or not (root / env["project"] / "pyproject.toml").is_file()
        ):
            errors.append(f"{name}: missing environment lock")
    return errors


def contract_digest(entry):
    contract = {
        k: v
        for k, v in entry.items()
        if k not in ["verificationRecord", "runtime", "lesson"]
    }
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


def render_lab(manifest):
    lines = [
        "---",
        "title: Run the first verified examples",
        "description: Trace attention, then run a pretrained sentence embedding model on your CPU.",
        "---",
        "",
        'import { Card, CardGrid } from "@astrojs/starlight/components";',
        "",
        "# From a calculation to a model",
        "",
        "Start with a visible calculation, then use pretrained weights to retrieve a sentence. Both examples run locally. The page shows source and output from recorded runs.",
        "",
        "<CardGrid>",
        '  <Card title="1 · Trace attention" icon="pencil">Invented scores → normalized weights → weighted value. CPU, offline, Python standard library.</Card>',
        '  <Card title="2 · Retrieve by meaning" icon="magnifier">Real pretrained model → 384-dimensional embeddings → cosine ranking. CPU; initial model download required.</Card>',
        "</CardGrid>",
        "",
        "## Set up",
        "",
        "Install [uv](https://docs.astral.sh/uv/getting-started/installation/), clone the repository, and run these commands from its root. `uv sync --locked` installs the exact resolved environment.",
        "",
        "```bash",
        "uv sync --locked --project examples/foundations",
        "uv sync --locked --project examples/applications",
        "python3 scripts/run-examples.py --id attention-softmax",
        "python3 scripts/run-examples.py --id embedding-retrieval --allow-network",
        "```",
        "",
        "The model is Apache-2.0 licensed. It is the 3-layer `sentence-transformers/paraphrase-MiniLM-L3-v2` checkpoint, with its exact revision recorded below. Its small fixture tests inference and ranking, with no training or quality benchmark claim.",
        "",
        "## Predict before running",
        "",
        "For scores 2 and 1, should the first value receive more than half the weight? For the query “A kitten is resting on a couch”, predict which of the three document sentences will rank first.",
        "",
    ]
    for eid, heading in [
        ("attention-softmax", "Trace the calculation"),
        ("embedding-retrieval", "Run the pretrained model"),
    ]:
        entry = next(e for e in manifest["examples"] if e["id"] == eid)
        lines += [f"## {heading}", "", "```python", region(entry), "```", ""]
        if entry["verificationRecord"]:
            record = json.loads((ROOT / entry["verificationRecord"]).read_text())
            lines += [
                "**Recorded output**",
                "",
                "```text",
                record["stdout"].rstrip(),
                "```",
                "",
                f"Verified {record['date']} on {record['platform']} / Python {record['python']}. Elapsed {record['durationSeconds']} s; peak child memory {record['peakMemoryMiB']} MiB. These measurements describe this run only.",
                "",
            ]
        lines += [
            f"[Source and run instructions](https://github.com/shadril238/llm-concept-atlas/tree/main/{manifest['environments'][entry['environment']]['project']})",
            "",
        ]
    lines += [
        "## Interpret and investigate",
        "",
        "Attention uses invented scores to isolate normalization; the transformer normally learns projections that produce these scores. Adding the same constant to both scores preserves the weights. Try scores 1002 and 1001 and explain why subtracting the maximum avoids overflow.",
        "",
        "The model converts token vectors into sentence vectors by averaging only positions allowed by the padding mask, then normalizes each vector. The dot product of normalized vectors is cosine similarity. It is a ranking score, not a probability. The run saves and reloads `embeddings.npy` as an artifact check.",
        "",
        "The five authored sentences form a demonstration fixture. The two correct rankings show that the wiring works for these cases. Add an ambiguous query and inspect all scores; this fixture cannot establish general retrieval quality.",
        "",
        "## Troubleshoot and continue",
        "",
        "A first model download needs internet access and roughly 70 MB of model weights plus tokenizer files. Package installation also downloads PyTorch. No account or API key is required. Later runs can use the cached snapshot with `--offline`; a missing cache causes a recorded failure. See the [example guide](https://github.com/shadril238/llm-concept-atlas/blob/main/examples/README.md) for environments, records, and troubleshooting.",
        "",
        "Return to [attention](/llm-concept-atlas/concepts/attention/) or [semantic search](/llm-concept-atlas/concepts/semantic-search/). Deeper lessons and training workflows follow in later enhancement phases.",
        "",
    ]
    return "\n".join(lines)
