"""Pedagogical notebook views of one authoritative source implementation."""

SPECS = {
    "instruction-adaptation": (
        "L21: instruction adaptation",
        "sft",
        "e8-sft",
        "Predict which tokens get a loss. Why must the real EOS survive when pad has the same token ID? Compare parameter counts before looking at task quality.",
        "print([(r['kind'],r['selected_step'],r['test_nll']) for r in result['runs']])\nprint(result['inspection'][0])",
        "Locate a full/LoRA failure and compare its reference/context. Explain trained-checkpoint selection versus choosing the unmodified baseline. Calculate effective batch size and LoRA parameters.",
    ),
    "preference-adaptation": (
        "L23: reward modeling and DPO",
        "preferences",
        "e8-preferences",
        "Predict the initial DPO loss with identical policy/reference. Why is the scalar reward head not used by DPO?",
        "print(result['reward']['test'])\nprint(result['dpo']['test'])",
        "Inspect heldout pair scores and length baselines. Explain raw preference accuracy versus reference-relative accuracy and ties. Trace causal shift and EOS through the shared source.",
    ),
    "generation-evaluation": (
        "L22: generation evaluation",
        "evaluate",
        "e8-evaluation",
        "Run L21 then L23 first. Predict which score prefers a wrong-day sentence over a valid paraphrase. Is exact mismatch sufficient evidence of factual error?",
        "print([(r['model'],r['exact_match'],r['rougeL'],r['bertscore_f1']) for r in result['runs']])\nprint(result['metric_probes'])",
        "Write a blinded human assessment without looking at model names. Import exact-output-bound labels with the offline importer. Explain why tiny-encoder BERTScore is not comparable with a different encoder.",
    ),
}


def notebook(id):
    title, module, artifact, predict, inspect, exercise = SPECS[id]

    def md(id, s):
        return dict(
            cell_type="markdown", id=id, metadata={}, source=s.splitlines(keepends=True)
        )

    def code(id, s):
        return dict(
            cell_type="code",
            id=id,
            metadata={},
            source=s.splitlines(keepends=True),
            execution_count=None,
            outputs=[],
        )

    setup = """import sys,json
from pathlib import Path
project=Path.cwd()
if project.name=='notebooks':project=project.parent
root=project.parents[2]
sys.path.insert(0,str(project/'src'))
manifest=json.loads((root/'examples/data/manifests/instruction-dolly.json').read_text())
print(manifest['source'],manifest['license'],manifest['split'])
"""
    cells = [
        md(
            "question",
            "# "
            + title
            + "\n\nData → objective → optimization → validation selection → heldout outputs → artifact reload.",
        ),
        md(
            "setup",
            "## Setup\n\nUse Python3.11: `uv sync --locked --project examples/training/generation`. Select that kernel and Restart/Run All. Pythia14m (~54MB weights), tiny BERT (~18MB), package/artifact memory additional. Float32 CPU, one thread; no API key. Read README. GPU QLoRA is unverified. Dataset subset retains CC BY-SA3.0 attribution.",
        ),
        code("provenance", setup),
        md("predict", "## Predict\n\n" + predict),
        code("train", f"from atlas_generation.{module} import main\nmain()"),
        md(
            "inspect",
            "## Inspect the saved results\n\nRead actual outputs, selection traces and failure categories. A lower training loss is not task success.",
        ),
        code(
            "inspect-results",
            f"result=json.loads((root/'examples/artifacts/{artifact}/results.json').read_text())\n"
            + inspect,
        ),
        md(
            "exercise",
            "## Transfer exercise\n\n"
            + exercise
            + "\n\nIndependent human/editorial review is pending. No benchmark or large-model generalization claim.",
        ),
        code(
            "reload-artifact",
            f"path=root/'examples/artifacts/{artifact}/results.json'\nassert json.loads(path.read_text())==result\nprint('Result artifact round trip verified')",
        ),
    ]
    return dict(
        nbformat=4,
        nbformat_minor=5,
        metadata=dict(
            kernelspec=dict(
                display_name="Python 3.11 (atlas generation)",
                language="python",
                name="python3",
            ),
            language_info=dict(name="python", version="3.11"),
        ),
        cells=cells,
    )
