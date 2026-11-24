"""L22: fixed task evaluation, real lexical/BERTScore metrics, offline judgments."""

import json
import math
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas_foundations.e8.contracts import output_hash, validate_judgments
from atlas_generation.common import ROOT, SPECS, download, save
from bert_score import BERTScorer
from rouge_score.rouge_scorer import RougeScorer
from sacrebleu.metrics import BLEU


def normalize(text):
    return " ".join(re.findall(r"\w+", text.lower()))


def score_rows(rows, scorer):
    rouge = RougeScorer(["rougeL"], use_stemmer=True)
    bleu = BLEU(effective_order=True)
    candidates = [r["prediction"] or " " for r in rows]
    refs = [r["reference"] for r in rows]
    precision, recall, f1 = scorer.score(candidates, refs, batch_size=4)
    out = []
    for r, p, rec, f in zip(rows, precision.tolist(), recall.tolist(), f1.tolist()):
        exact = normalize(r["prediction"]) == normalize(r["reference"])
        error = (
            "exact"
            if exact
            else "empty"
            if not r["prediction"].strip()
            else "generation-limit"
            if r.get("hit_limit")
            else "reference-mismatch"
        )
        out.append(
            dict(
                r,
                exact_match=exact,
                rougeL=rouge.score(r["reference"], r["prediction"])["rougeL"].fmeasure,
                bertscore_precision=p,
                bertscore_recall=rec,
                bertscore_f1=f,
                error_category=error,
                output_sha256=output_hash(r["prediction"]),
            )
        )
    return dict(
        exact_match=sum(r["exact_match"] for r in out) / len(out),
        rougeL=statistics.mean(r["rougeL"] for r in out),
        bleu=bleu.corpus_score([r["prediction"] for r in rows], [refs]).score,
        bleu_signature=str(bleu.get_signature()),
        bertscore_f1=statistics.mean(r["bertscore_f1"] for r in out),
        denominator=len(rows),
        rows=out,
    )


def main():
    sft = json.loads((ROOT / "examples/artifacts/e8-sft/results.json").read_text())
    pref = json.loads(
        (ROOT / "examples/artifacts/e8-preferences/results.json").read_text()
    )
    scorer = BERTScorer(
        model_type=download("prajjwal1/bert-tiny"),
        num_layers=2,
        device="cpu",
        idf=False,
        rescale_with_baseline=False,
        lang="en",
    )
    runs = []
    for name, nll_value, rows in (
        [("base", sft["baseline"]["nll"], sft["baseline"]["generations"])]
        + [(r["kind"], r["test_nll"], r["generations"]) for r in sft["runs"]]
        + [("dpo", pref["dpo"]["test_nll"], pref["dpo"]["generations"])]
    ):
        scored = score_rows(rows, scorer)
        scored.update(
            model=name,
            completion_nll=nll_value,
            completion_perplexity=math.exp(nll_value),
        )
        runs.append(scored)
    # Authored contrasts diagnose metrics; never mixed into heldout model results.
    reference = "The package arrived on Monday."
    probes = score_rows(
        [
            dict(
                id="paraphrase",
                prediction="The parcel was delivered on Monday.",
                reference=reference,
            ),
            dict(
                id="wrong-day",
                prediction="The package arrived on Tuesday.",
                reference=reference,
            ),
            dict(id="exact", prediction=reference, reference=reference),
        ],
        scorer,
    )
    blind = []
    for run in runs:
        for r in run["rows"]:
            blind.append(
                dict(model=run["model"], id=r["id"], prediction=r["prediction"])
            )
    imported = []
    path = ROOT / "examples/data/local/generation-judgments.json"
    if path.exists():
        imported = json.loads(path.read_text())
    judgments = validate_judgments(blind, imported)
    result = dict(
        runs=runs,
        metric_probes=probes,
        semantic_metric=dict(
            model="prajjwal1/bert-tiny",
            revision=SPECS["prajjwal1/bert-tiny"]["revision"],
            layers=2,
            idf=False,
            rescale_with_baseline=False,
            library="bert-score 0.3.13",
            warning="Tiny demonstration encoder, not standard RoBERTa-large BERTScore; scores not comparable across encoders or settings.",
        ),
        judgments=judgments,
        rubric=dict(
            version="atlas-e8-v1",
            grounding="0 contradicted/unsupported, 1 partially supported, 2 all claims supported by context",
            instruction="0 wrong task, 1 partial format/content, 2 all requirements met",
            clarity="0 unintelligible, 1 understandable with noise, 2 concise and clear",
        ),
        human_review="Pending; no ratings fabricated. Blind model identities, randomize order, include ties and adjudication; never use test judgments for tuning.",
        judge_workflow="Optional offline import only: record judge model/revision, prompt and rubric version; AB/BA order, output SHA; calibrate on authored exact/wrong-day/paraphrase probes and human labels before interpreting agreement.",
        limits="Single-reference extraction on eight heldout cases. Exact mismatch is not necessarily factual error. BLEU/ROUGE lexical; BERTScore contextual token matching; none alone verifies truth.",
    )
    save("e8-evaluation", result)
    print(
        json.dumps(
            dict(
                fixed_suite_verified=len(runs) == 4
                and all(r["denominator"] == 8 for r in runs),
                semantic_metric_executed=True,
                judge_import_verified=True,
            )
        )
    )
    return result


if __name__ == "__main__":
    main()
