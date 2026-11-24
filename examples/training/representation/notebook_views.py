"""Staged notebooks import the actual shared training modules, without copied algorithms."""

import json

SPECS = {
    "embedding-training": (
        "L15 · losses and supervised embedding training",
        "embedding",
        "e7-embedding",
        "Predict which objective uses binary relation labels, target cosine values, or in-batch alternatives. Inspect reserved split IDs and the reserve/cancel near miss.",
        'print("Baseline:",result["baseline"]["retrieval"]["mrr"])\nprint([(r["kind"],r["retrieval"]["mrr"],r["similarity"]["mse"])for r in result["conditions"]])',
        "Change a batch so two candidates share the same intent. Explain why a lower cross-entropy can reward a false distinction. Compare full-dimensional quality with the fixed two-dimensional PCA display.",
    ),
    "augmented-sbert-training": (
        "L16 · train the teacher and inspect silver errors",
        "augmentation",
        "e7-augmentation",
        "Predict whether adding 120 imperfect teacher labels must improve the student. Candidate pairs are sampled without latent intent labels from a separate training-only pool.",
        'print("Teacher:",result["teacher"]["test"])\nprint("Silver disagreements:",sum(r["disagreement"]for r in result["diagnostics"]))\nprint([(r["condition"],r["similarity"]["mse"])for r in result["students"]])',
        "Find a teacher disagreement and trace its pair IDs into the training pool. Explain why equal step budgets do not equal identical gold exposure. Design a fresh validation-only filtering experiment without selecting on this test set.",
    ),
    "tsdae-adaptation": (
        "L17 · reconstruction versus useful embeddings",
        "denoising",
        "e7-denoising",
        "Predict what information deletion removes and what the causal decoder can infer from a single CLS vector. Teacher forcing and greedy reconstruction expose different evidence.",
        'print("Actual reconstruction:",result["reconstruction"])\nprint("Disagreement intervals:",result["counterexamples"])\nprint("Before/after retrieval:",result["baseline"]["retrieval"]["mrr"],result["tsdae"]["retrieval"]["mrr"])',
        "Find an interval in which reconstruction loss decreases without retrieval improvement. Explain tied encoder/decoder parameters and the separate newly initialized cross-attention. Compare supervised adaptation with and without TSDAE.",
    ),
    "representation-classification": (
        "L18 · frozen layers and repeated few-shot SetFit",
        "classification",
        "e7-classification",
        "The fixed 200/40/80 review splits are reused from E4. Three seeds each select eight examples per class; every method receives identical selected IDs within a seed. Compare full, partial and head-only BERT, lexical and frozen sentence baselines, and real SetFit.",
        'print(json.dumps(result["summary"],indent=2))\nprint("First freeze report:",result["runs"][0]["parameters"])',
        "Find a seed where adapting the sentence encoder underperforms the frozen encoder. Explain why pair multiplication does not increase the number of independent labels. Distinguish BERT/MiniLM architecture differences from a controlled objective comparison.",
    ),
    "continued-pretraining": (
        "L19 · actual masking, adaptation and downstream evaluation",
        "mlm",
        "e7-mlm",
        "Predict which tokens receive loss and why selected unchanged tokens can still be targets. The two-step smoke profile is separate from the sixty-step evaluated run; no test review enters adaptation.",
        'print("Mask trace:",result["mask_trace"])\nprint([(r["condition"],r["test"]["macro_f1"])for r in result["downstream"]])\nassert result["downstream"][0]["head_pooler_initial_sha256"]==result["downstream"][1]["head_pooler_initial_sha256"]',
        "Explain why an MLM checkpoint that omits the pooler could confound the classifier comparison. Inspect the explicit shared-weight transfer and equal initial pooler/head hashes. Explain why a fill-mask completion is insufficient downstream evidence.",
    ),
    "ner-complete": (
        "L20 · complete word/subword alignment and entity evaluation",
        "ner",
        "e7-ner",
        "Reuse the E2 trainer rather than duplicating it. Compare first-subtoken masking and B-to-I propagation, safe truncation rejection, padding, multiword entities and no-entity input.",
        'print("Real entity scores:",result["training"]["metrics"])\nprint("Alignment:",result["alignment"][0])\nprint("Span mistakes:",result["span_examples"])',
        "Correct the boundary and type examples under strict sentence-preserving span scoring. Explain the explicit invalid-I repair policy and how it differs from silently accepting malformed gold BIO. State why this small templated corpus is not a natural-domain NER benchmark.",
    ),
}


def notebook(id):
    title, module, artifact, predict, inspect, exercise = SPECS[id]

    def md(id, text):
        return dict(
            cell_type="markdown",
            id=id,
            metadata={},
            source=text.splitlines(keepends=True),
        )

    def code(id, text):
        return dict(
            cell_type="code",
            id=id,
            metadata={},
            execution_count=None,
            outputs=[],
            source=text.splitlines(keepends=True),
        )

    setup = """import sys,json
from pathlib import Path
project=Path.cwd()
if project.name=='notebooks':project=project.parent
root=project.parents[2]
sys.path.insert(0,str(project/'src'))
fixture=json.loads((root/'examples/data/fixtures/embedding-training.json').read_text())
from collections import Counter
print('Original embedding splits:',Counter(r['split']for r in fixture['rows']))
print('Original policy:',fixture['policy'])
print('Review data:',json.loads((root/'examples/data/manifests/sentiment.json').read_text())['license'])
"""
    cells = [
        md(
            "question",
            "# "
            + title
            + "\n\nFollow data → objective → gradients → checkpoint → held-out predictions → reusable artifact. No API or paid provider. CPU training is intentionally small, with original or licensed data and visible failures.\n",
        ),
        md(
            "setup",
            "## Set up\n\nRun `uv sync --locked --project examples/training/representation` and select that Python 3.11 kernel. Start in this project or its notebooks directory. The pinned tiny BERT weights are about 17.8 MB; actual SetFit additionally uses about 70 MB of pretrained MiniLM sentence weights. First downloads need network; later use HF_HUB_OFFLINE=1. CPU/float32/one thread is the verified profile; GPU runs are not claimed. Read README for measured resources and exact package/data/model pins. Restart and Run All.\n",
        ),
        code("inspect-data", setup),
        md(
            "prediction",
            "## Predict before training\n\n"
            + predict
            + "\n\nThe next cell imports the actual training implementation. It initializes a fresh model, performs real optimization and validation selection, checks intended gradients/updates and frozen parameters where applicable, then saves/reloads model artifacts.\n",
        ),
        code("train", f"from atlas_representation.{module} import main\nmain()"),
        md(
            "inspect",
            "## Inspect real results and errors\n\nTask metrics, loss curves, source IDs, actual predictions and limits are retained. The test set never chooses a checkpoint or tunes a hyperparameter. A poor result is still valid execution evidence.\n",
        ),
        code(
            "results",
            f"artifact_path=root/'examples/artifacts/{artifact}/results.json'\nresult=json.loads(artifact_path.read_text())\n{inspect}",
        ),
        md(
            "work",
            "## Account for work and reproducibility\n\nCompare optimizer steps, batches, selected checkpoint, model calls, trainable counts, float32 CPU placement and saved artifact hashes. A small fixed-step run is not a broad benchmark or convergence claim. SetFit uses fixed training steps; the BERT/embedding paths select from declared validation checkpoints. Independent review and beginner feedback remain pending.\n",
        ),
        code(
            "reload",
            "assert json.loads(artifact_path.read_text())==result\nprint('Saved result reload checked; the source also checks actual model reload equivalence.')",
        ),
        md(
            "transfer",
            "## Transfer exercise\n\n"
            + exercise
            + "\n\nReport: data IDs and provenance; baseline; changed component; prediction; actual learning curve; frozen/update checks; selected checkpoint; untouched evaluation; representative failure; runtime; artifact reload; next fresh-data test.\n\n### Solution guidance\n\nUse source IDs and tensor/mask boundaries before interpreting a score. Do not relabel test data as training-only, silver as gold, dependent pairs as new independent labels, a projection as full geometry, or a surrogate loss decrease as task success.\n",
        ),
        md(
            "boundary",
            "## Verification boundary\n\nAll code cells run in a fresh actual kernel. Notebooks and CLI import one implementation. Run notebooks before final CLI --record commands because they overwrite timing-bearing artifacts. Original prose/code are MIT, licensed review data retain attribution, and book pages/artwork are not redistributed.\n",
        ),
    ]
    return dict(
        nbformat=4,
        nbformat_minor=5,
        metadata=dict(
            kernelspec=dict(
                display_name="Python 3.11 (atlas representation)",
                language="python",
                name="python3",
            ),
            language_info=dict(name="python", version="3.11"),
        ),
        cells=cells,
    )
