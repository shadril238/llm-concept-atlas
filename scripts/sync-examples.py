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
    ("reference", "first-generation"),
    ("reference", "representation-inspection"),
    ("reference", "playlist-embeddings"),
    ("reference", "transparent-transformer"),
    ("reference", "cache-equivalence"),
    ("workflows", "prompt-evaluation"),
    ("workflows", "search-and-voting"),
    ("workflows", "local-model-chain"),
    ("workflows", "memory-policies"),
    ("workflows", "bounded-tool-workflow"),
    ("discovery", "classification-comparison"),
    ("discovery", "modular-bertopic"),
    ("discovery", "topic-label-generation"),
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
        "first-generation": "atlas_reference.e3.generation",
        "representation-inspection": "atlas_reference.e3.representations",
        "playlist-embeddings": "atlas_reference.e3.playlists",
        "transparent-transformer": "atlas_reference.e3.transformer",
        "cache-equivalence": "atlas_reference.e3.cache",
        "prompt-evaluation": "atlas_workflows.prompts",
        "search-and-voting": "atlas_workflows.search_vote",
        "local-model-chain": "atlas_workflows.chains",
        "memory-policies": "atlas_workflows.memory",
        "bounded-tool-workflow": "atlas_workflows.agents",
        "classification-comparison": "atlas_discovery.classification",
        "modular-bertopic": "atlas_discovery.topics",
        "topic-label-generation": "atlas_discovery.topic_labels",
    }[eid]
    # Generated notebooks import the same source used by the CLI; no copied algorithm.
    code = f'import sys\nfrom pathlib import Path\nproject = Path.cwd()\nif project.name == "notebooks": project = project.parent\nsys.path.insert(0, str(project / "src"))\nfrom {module} import main\nmain()'
    if eid == "playlist-embeddings":
        code = code.rsplit("main()", 1)[0] + "main([])"
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
    if family == "workflows":
        spec = {
            "prompt-evaluation": (
                "L09: can prompt changes improve task answers or only formatting?",
                "e5-prompts",
                "Compare six arithmetic and three extraction conditions on the same held-out cases. Predict which examples might encourage a repeated-answer shortcut. Native-schema and instruction-only extraction have identical messages.",
                "print(json.dumps(result['summary'], indent=2))\nprint('Truncated native prefix:', repr(result['truncation_probe']['raw']))",
                "Find a valid but incorrect order. Explain which semantic check rejects it. Why is strict extracted-answer accuracy different from human interpretation of prose?",
                "A schema restricts shape, not the source fact. Retain invalids and report denominators. Native truncation can leave only an allowed prefix.",
            ),
            "search-and-voting": (
                "L10: when do voting and search fail?",
                "e5-search",
                "Predict what happens to a fractional Game-24 route under beam one versus complete search. The two sampling paths change instruction and decoding, so compare them as labeled mechanism probes, not an isolated intervention.",
                "print(json.dumps([{'case':r['id'],'path':r['variant'],'vote':r['aggregation']} for r in result['samples']], indent=2))\nprint(json.dumps([{k:r[k] for k in ['id','strategy','stop_reason','solution','created','expanded']} for r in result['games']], indent=2))",
                "Inspect 3,3,8,8. Does a node cap prove impossibility? Can a unique plurality have fewer than half of all samples?",
                "8/(3-8/3) is an exact solution. Pruning/budget failures do not exhaust the legal frontier. Ties abstain and invalid samples remain in the strict-majority denominator.",
            ),
            "local-model-chain": (
                "L11a: where does an error enter a real chain?",
                "e5-chains",
                "Predict whether equivalent typed-message and plain-Python rendering must produce the same actual prompt. Then trace generated extraction through deterministic pricing into the second actual prompt.",
                "print('Adapter checks:', result['equivalent_prompt_checked'], result['equivalent_output_checked'])\nchain=result['multi_prompt']\nprint(json.dumps({k:chain[k] for k in ['state','final','expected_total','extraction_correct','copy_consistent','end_to_end_correct']}, indent=2))",
                "If the second model copies a supplied total correctly, has the original request necessarily been answered?",
                "No. Check extraction against the original request before computing/copying a total. Internal consistency can coexist with end-to-end failure.",
            ),
            "memory-policies": (
                "L11b: which facts reach and influence the next answer?",
                "e5-memory",
                "Use the identical eight-message conversation for buffer, two-pair window, and actual running summary. Predict where the old code disappears and how the Monday correction should supersede Friday. Budgets include output reserve.",
                "print(json.dumps([{'policy':r['policy'],'budget':r['budget'],'packing':r['packed'],'field_correct':r['field_correct']} for r in result['cases']], indent=2))\nprint('Raw summary updates:', [u['summary'] for u in result['summary_updates']])",
                "Distinguish policy omission, budget eviction, summary distortion, and model failure to use a retained fact. Which happened to the code?",
                "Use message IDs and actual generated summary text. Literal presence is only a diagnostic; grade final fields independently. Summary updates add model work.",
            ),
            "bounded-tool-workflow": (
                "L11c: can a tool workflow terminate and still be wrong?",
                "e5-agents",
                "Inspect the finite tool and final-value domains. Predict how invalid booleans, repeated actions, and a wrong tool choice should appear. Scripted/FakeListLLM tests measure mechanics, separately from actual GGUF choices.",
                "print(json.dumps([{'id':r['id'],'engine':r['engine'],'expected':r['expected'],'correct':r['correct'],'outcome':r['outcome']} for r in result['real_cases']], indent=2))\nprint('Fake-model stop reasons:', [(r['id'], r['framework']['stop_reason']) for r in result['fake_cases']])",
                "Does max_steps count as task success? Does a valid action ID prove the chosen tool is appropriate?",
                "Neither establishes an answer. The controller guarantees a bounded path, while source-based correctness and tool selection are separately evaluated. Operands are predeclared; this is a restricted controller.",
            ),
        }[eid]
        question, artifact, baseline, inspect_code, exercise, solution = spec

        def md(id, text):
            return {
                "cell_type": "markdown",
                "id": id,
                "metadata": {},
                "source": text.splitlines(keepends=True),
            }

        def cell(id, text):
            return {
                "cell_type": "code",
                "id": id,
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": text.splitlines(keepends=True),
            }

        setup = """import sys, json
from pathlib import Path
project = Path.cwd()
if project.name == "notebooks": project = project.parent
root = project.parents[2]
sys.path.insert(0, str(project / "src"))
fixture = json.loads((root / "examples/data/fixtures/workflow-tasks.json").read_text())
print("Original fixture fields:", list(fixture))
for name in ["arithmetic", "extraction"]:
    rows = fixture[name]
    print(name, {split: sum(r["split"] == split for r in rows) for split in ["design", "evaluation"]})
    print("Design example:", next(r for r in rows if r["split"] == "design"))
"""
        notebook["cells"] = [
            md(
                "question",
                f"# {question}\n\nRead the connected lesson and predict before running. No API key or hosted provider is used. All original fixture data is MIT. This notebook imports the shared tested source; no copied model algorithm.\n",
            ),
            md(
                "execution-contract",
                "## Execution contract\n\nRun `uv sync --locked --project examples/applications/workflows`, select that environment's Python kernel, and start in this project or its notebooks directory. Pinned Q8 SmolLM2 GGUF: 144,811,360-byte first download, SHA-256 checked, CPU computation, native context 768, one thread. Installation compiles llama-cpp-python and needs a compiler/CMake. Package/build downloads are additional.\n\nOn the author host each script took about 1–10 seconds after setup; hardware and prefix caching affect timing. Recorded child process memory is roughly 0.5–0.6 GiB, not a precision-comparison benchmark. On macOS the native build initializes Metal even with CPU buffers; use a normal host runtime if a managed sandbox denies initialization. See README for exact pins and provenance. Restart the kernel and Run All; the first run needs network for the public artifact, later runs can set HF_HUB_OFFLINE=1.\n",
            ),
            md(
                "inspect-data",
                "## Inspect data before inference\n\nThese original tasks have stable IDs. Arithmetic has 4 design / 8 evaluation cases; extraction has 3 design / 9 evaluation cases. Memory/search/tools are declared mechanism probes. No training or evaluation-set tuning occurs. Demonstrations use design IDs only; gold answers are not inserted into per-case decoding domains.\n",
            ),
            cell("inspect-fixture", setup),
            md(
                "baseline",
                "## Predict and inspect the baseline\n\n"
                + baseline
                + "\n\nRead the module source linked in the lesson before invoking it. It includes the baseline and declared alternatives in one bounded run. A result need not improve.\n",
            ),
            cell("run-source", f"from {module} import main\nmain()"),
            md(
                "evaluate",
                "## Evaluate and inspect failures\n\nRead the actual artifact produced above. Shape validity, semantic correctness, retained evidence, and controller termination answer different questions. Raw calls retain messages, native token counts, output caps, stop reasons, and unedited output. The tiny model's failures remain part of the evidence.\n",
            ),
            cell(
                "inspect-result",
                f"artifact_path = root / 'examples/artifacts/{artifact}/results.json'\nresult = json.loads(artifact_path.read_text())\n{inspect_code}",
            ),
            md(
                "work-accounting",
                "## Work and controlled comparisons\n\nPrompt/input and generated/output tokens are recorded per call. Summary updates, repeated samples, and model-guided ranking add calls. Single sequential wall timings reuse a loaded model and prefix cache; do not treat them as a speed leaderboard, paid cost, or causal comparison across unrelated paths. The declared comparison changes only its named component except where explicitly labeled as a separate mechanism probe.\n",
            ),
            cell(
                "artifact-reload",
                "# Reload the saved result, preserving its link to actual inputs/settings.\nreloaded = json.loads(artifact_path.read_text())\nassert reloaded == result\nprint('Saved artifact reload checked:', artifact_path.name)\nprint('Model artifact revision:', result['model']['revision'])",
            ),
            md(
                "transfer",
                f"## Practice and transfer\n\n{exercise}\n\n<details><summary>Hint and solution rationale</summary>\n\n{solution}\n\n</details>\n\nFor a transfer experiment, predeclare a new design/evaluation fixture and change one component. Do not select a best prompt or seed on these evaluation outcomes. Report: task IDs; baseline; changed component; expected failure; raw evidence; parser/verifier; denominators; tokens/calls; observed limitations. Broader model, device, or unrestricted-agent claims need new evidence.\n",
            ),
            md(
                "verification-boundary",
                "## Verification boundary\n\nThe notebook and CLI call the same source. Run notebook verification before final `--record` CLI runs because timing-bearing artifacts are overwritten. Then sync published results. Execution and author review do not establish independent teaching acceptance or beginner usability. No source PDF or copied artwork is distributed.\n",
            ),
        ]

    outputs[f"{manifest['environments'][family]['project']}/notebooks/{eid}.ipynb"] = (
        json.dumps(notebook, indent=2) + "\n"
    )
hosted_notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3.11 (offline importer)",
            "language": "python",
            "name": "python3",
        }
    },
    "cells": [
        {
            "cell_type": "markdown",
            "id": "workflow",
            "metadata": {},
            "source": [
                "# Optional hosted classification: collect separately, evaluate offline\n",
                "This notebook makes **zero API requests**. Read the E4 README before a separate provider run. Freeze the 80 test IDs, label policy, prompt/model revision, output-token cap, and call/retry limit. Record refusals/errors/missing outputs, real token usage/cost/latency only when measured, and retain source IDs. Store the collected envelope at repository `.cache/hosted-e4-results.json`. Never put credentials in this notebook or the static site.\n",
                "Required envelope fields: provider, model, prompt_version, collected_at, outputs. Each output has id, raw (plain label or JSON string with exactly label), optional error/latency_seconds. This importer rejects unknown/duplicate IDs and includes missing/error outputs as invalid predictions. External provider execution is not verified by the atlas.\n",
            ],
        },
        {
            "cell_type": "code",
            "id": "import-offline",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "import sys, json\n",
                "from pathlib import Path\n",
                "project = Path.cwd()\n",
                "if project.name == 'notebooks': project = project.parent\n",
                "root = project.parents[2]\n",
                "sys.path.insert(0, str(root / 'examples/foundations/src'))\n",
                "from atlas_foundations.e4.hosted_results import evaluate_external\n",
                "fixture = json.loads((root / 'examples/data/fixtures/sentiment.json').read_text())\n",
                "collected = root / '.cache/hosted-e4-results.json'\n",
                "if collected.exists():\n",
                "    report = evaluate_external(fixture, json.loads(collected.read_text()))\n",
                "    print(json.dumps(report, indent=2))\n",
                "else:\n",
                "    print('No hosted results supplied; no API was called and no hosted metrics are claimed.')\n",
                "    print('Required test IDs:', [r['id'] for r in fixture['rows'] if r['split'] == 'test'])\n",
            ],
        },
    ],
}
outputs["examples/applications/discovery/notebooks/hosted-comparison.ipynb"] = (
    json.dumps(hosted_notebook, indent=2) + "\n"
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
if not args.check:
    names = ["generation", "representations", "playlists", "transformer", "cache"]
    paths = {
        name: ROOT / "examples/artifacts" / f"e3-{name}" / "results.json"
        for name in names
    }
    if all(path.exists() for path in paths.values()):
        outputs["src/data/foundation-results.json"] = (
            json.dumps(
                {name: json.loads(path.read_text()) for name, path in paths.items()},
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
if not args.check:
    workflow_paths = {
        key: ROOT / "examples/artifacts" / artifact / "results.json"
        for key, artifact in {
            "prompts": "e5-prompts",
            "search": "e5-search",
            "chains": "e5-chains",
            "memory": "e5-memory",
            "agents": "e5-agents",
        }.items()
    }
    if all(path.exists() for path in workflow_paths.values()):
        outputs["src/data/workflow-results.json"] = (
            json.dumps(
                {
                    key: json.loads(path.read_text())
                    for key, path in workflow_paths.items()
                },
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
# Replace only explicit source regions; surrounding explanations stay authored.
if not args.check:
    discovery = {
        "classification": "e4-classification",
        "topics": "e4-topics",
        "generation": "e4-topic-labels",
    }
    paths = {
        key: ROOT / "examples/artifacts" / name / "results.json"
        for key, name in discovery.items()
    }
    if all(path.exists() for path in paths.values()):
        outputs["src/data/discovery-results.json"] = (
            json.dumps(
                {key: json.loads(path.read_text()) for key, path in paths.items()},
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
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
