"""L07: same-case comparison of seven actual classification methods."""

import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas_discovery.common import (
    ROOT,
    Encoder,
    NLI,
    NLI_REV,
    SENTIMENT,
    SENTIMENT_REV,
    T5,
    T5_REV,
    measure,
    save_result,
    setup,
)
from atlas_foundations.e4.metrics import (
    classification_metrics,
    parse_label,
    selective_metrics,
)

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


def probability_rows(scores):
    scores = np.asarray(scores)
    assert np.isfinite(scores).all() and np.allclose(scores.sum(1), 1, atol=1e-5)
    return [
        {
            "prediction": int(row.argmax()),
            "confidence": float(row.max()),
            "scores": row.tolist(),
        }
        for row in scores
    ]


def main():
    setup()
    data = json.loads((ROOT / "examples/data/fixtures/sentiment.json").read_text())
    train = [r for r in data["rows"] if r["split"] == "train"]
    valid = [r for r in data["rows"] if r["split"] == "validation"]
    held = [r for r in data["rows"] if r["split"] in ["test", "challenge"]]
    assert len({r["id"] for r in data["rows"]}) == len(data["rows"])
    text = lambda rows: [r["text"] for r in rows]
    labels = lambda rows: [r["label"] for r in rows]
    methods = {}

    def evaluate(name, predict, contract, training_seconds=0.0):
        validation_predictions = predict(valid)
        predictions, timing = measure(predict, held)
        summary = {
            "contract": contract,
            "fit_seconds": training_seconds,
            "timing": timing,
            "validation": classification_metrics(
                labels(valid), [r["prediction"] for r in validation_predictions]
            ),
            "cases": [
                dict(id=row["id"], **prediction)
                for row, prediction in zip(held, predictions)
            ],
        }
        for split in ["test", "challenge"]:
            positions = [i for i, row in enumerate(held) if row["split"] == split]
            summary[split] = classification_metrics(
                [held[i]["label"] for i in positions],
                [predictions[i]["prediction"] for i in positions],
            )
        methods[name] = summary
        return validation_predictions, predictions

    def fit_head(features, validation_features):
        candidates = []
        for regularization in [0.1, 1.0, 10.0]:
            head = LogisticRegression(
                C=regularization, max_iter=1000, random_state=37
            ).fit(features, labels(train))
            score = classification_metrics(
                labels(valid), head.predict(validation_features).tolist()
            )["macro_f1"]
            candidates.append((score, regularization, head))
        best = max(candidates, key=lambda item: (item[0], -item[1]))
        return best[2], [
            {"C": c, "validation_macro_f1": score} for score, c, _ in candidates
        ]

    # Fit vocabulary and IDF on training rows only, never all documents.
    start = time.perf_counter()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=8000)
    bow = vectorizer.fit_transform(text(train))
    baseline, baseline_candidates = fit_head(bow, vectorizer.transform(text(valid)))
    evaluate(
        "tfidf-logistic",
        lambda rows: probability_rows(
            baseline.predict_proba(vectorizer.transform(text(rows)))
        ),
        "Train-only TF-IDF features + fitted logistic head; C selected on validation macro-F1.",
        time.perf_counter() - start,
    )

    encoder = Encoder()
    start = time.perf_counter()
    train_vectors, valid_vectors = (
        encoder.encode(text(train)),
        encoder.encode(text(valid)),
    )
    head, embedding_candidates = fit_head(train_vectors, valid_vectors)
    fit_seconds = time.perf_counter() - start
    head_predict = lambda rows: probability_rows(
        head.predict_proba(encoder.encode(text(rows)))
    )
    val_predictions, held_predictions = evaluate(
        "embedding-logistic",
        head_predict,
        "Frozen 384D MiniLM vectors + train-only logistic head; C selected on validation macro-F1.",
        fit_seconds,
    )
    means = np.stack(
        [train_vectors[np.array(labels(train)) == label].mean(0) for label in [0, 1]]
    )
    means /= np.linalg.norm(means, axis=1, keepdims=True)

    def similarity_predict(rows, references):
        scores = encoder.encode(text(rows)) @ references.T
        assert np.isfinite(scores).all() and np.abs(scores).max() <= 1.00001
        return [
            {"prediction": int(r.argmax()), "confidence": None, "scores": r.tolist()}
            for r in scores
        ]

    evaluate(
        "class-prototypes",
        lambda rows: similarity_predict(rows, means),
        "Cosine to normalized mean training vectors per class. Uses training labels; this is supervised, not zero-shot.",
    )
    descriptions = [
        "A negative movie review expressing disappointment and dislike.",
        "A positive movie review expressing enjoyment and praise.",
    ]
    references = encoder.encode(descriptions)
    evaluate(
        "description-similarity",
        lambda rows: similarity_predict(rows, references),
        "Cosine to fixed label-description vectors. Scores are similarities, not calibrated probabilities; no local example labels used.",
    )
    alternate = encoder.encode(
        ["The viewer disliked this film.", "The viewer liked this film."]
    )
    wording_predictions = similarity_predict(held, alternate)
    wording_changes = [
        row["id"]
        for row, a, b in zip(
            held, methods["description-similarity"]["cases"], wording_predictions
        )
        if a["prediction"] != b["prediction"]
    ]

    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT, revision=SENTIMENT_REV)
    task_model = AutoModelForSequenceClassification.from_pretrained(
        SENTIMENT, revision=SENTIMENT_REV
    ).eval()
    mapping = {int(k): parse_label(v) for k, v in task_model.config.id2label.items()}
    assert sorted(mapping.values()) == [0, 1]

    def task_predict(rows):
        outputs = []
        for start in range(0, len(rows), 16):
            batch = tokenizer(
                text(rows[start : start + 16]),
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            with torch.inference_mode():
                probabilities = task_model(**batch).logits.softmax(-1).numpy()
            canonical = np.zeros_like(probabilities)
            for model_id, label in mapping.items():
                canonical[:, label] = probabilities[:, model_id]
            outputs.extend(probability_rows(canonical))
        return outputs

    evaluate(
        "task-encoder",
        task_predict,
        "Frozen SST-2 DistilBERT classification head; config.id2label explicitly mapped to atlas 0=negative/1=positive. No local fitting.",
    )
    nli_tokenizer = AutoTokenizer.from_pretrained(NLI, revision=NLI_REV)
    nli_model = AutoModelForSequenceClassification.from_pretrained(
        NLI, revision=NLI_REV
    ).eval()
    entailment = next(
        int(k)
        for k, v in nli_model.config.id2label.items()
        if v.lower() == "entailment"
    )
    hypotheses = [
        "This movie review expresses negative sentiment.",
        "This movie review expresses positive sentiment.",
    ]

    def nli_predict(rows):
        outputs = []
        for start in range(0, len(rows), 16):
            texts = text(rows[start : start + 16])
            batch = nli_tokenizer(
                [s for s in texts for _ in hypotheses],
                hypotheses * len(texts),
                padding=True,
                truncation="only_first",
                max_length=256,
                return_tensors="pt",
            )
            with torch.inference_mode():
                logits = nli_model(**batch).logits[:, entailment].reshape(-1, 2)
                outputs.extend(probability_rows(logits.softmax(-1).numpy()))
        return outputs

    evaluate(
        "nli-zero-shot",
        nli_predict,
        "Pair each review with two hypotheses; softmax of entailment logits across the mutually exclusive candidates. This is relative candidate scoring, not a three-way NLI probability or calibrated confidence.",
    )
    t5_tokenizer = AutoTokenizer.from_pretrained(T5, revision=T5_REV)
    generator = AutoModelForSeq2SeqLM.from_pretrained(T5, revision=T5_REV).eval()
    prompt = "Is the following movie review positive or negative? Answer with exactly positive or negative. Review: "

    def t5_predict(rows):
        outputs = []
        for start in range(0, len(rows), 16):
            batch = t5_tokenizer(
                [prompt + s for s in text(rows[start : start + 16])],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            with torch.inference_mode():
                tokens = generator.generate(**batch, max_new_tokens=8, do_sample=False)
            outputs.extend(
                {
                    "prediction": parse_label(raw),
                    "confidence": None,
                    "scores": None,
                    "raw": raw,
                }
                for raw in t5_tokenizer.batch_decode(tokens, skip_special_tokens=True)
            )
        return outputs

    evaluate(
        "flan-t5",
        t5_predict,
        "Greedy encoder-decoder label generation; max 8 new tokens. Trim/case-fold then exact positive/negative lookup; every other output is invalid=-1 and counts as an error.",
    )

    # Select a review threshold using validation only. Maximize retained coverage
    # subject to >=90% validation accuracy; if none qualifies, review every case.
    thresholds = []
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]:
        thresholds.append(
            dict(
                threshold=threshold,
                **selective_metrics(
                    labels(valid),
                    [p["prediction"] for p in val_predictions],
                    [p["confidence"] for p in val_predictions],
                    threshold,
                ),
            )
        )
    feasible = [
        r
        for r in thresholds
        if r["accepted_accuracy"] is not None and r["accepted_accuracy"] >= 0.9
    ]
    selected_threshold = (
        max(feasible, key=lambda r: (r["coverage"], -r["threshold"]))["threshold"]
        if feasible
        else 1.0
    )

    # Portable JSON inference artifact, no pickled executable estimator required.
    out = ROOT / "examples/artifacts/e4-classification"
    out.mkdir(parents=True, exist_ok=True)
    saved = {
        "always_review": not bool(feasible),
        "classes": head.classes_.tolist(),
        "coef": head.coef_.tolist(),
        "intercept": head.intercept_.tolist(),
        "C": head.C,
        "embedding_model": "sentence-transformers/paraphrase-MiniLM-L3-v2",
        "embedding_revision": "4ca70771034acceecb2e72475f72050fcdde4ddc",
        "review_threshold": selected_threshold,
        "max_length": 256,
    }
    (out / "classifier.json").write_text(json.dumps(saved, indent=2) + "\n")
    loaded = json.loads((out / "classifier.json").read_text())
    check_vectors = encoder.encode(text(held))
    restored_logits = check_vectors @ np.asarray(loaded["coef"]).T + np.asarray(
        loaded["intercept"]
    )
    restored_probability = 1 / (1 + np.exp(-restored_logits[:, 0]))
    assert np.allclose(
        restored_probability, head.predict_proba(check_vectors)[:, 1], atol=1e-10
    )
    result = {
        "fixture": data["id"],
        "label_map": data["labels"],
        "split_counts": {
            s: sum(r["split"] == s for r in data["rows"])
            for s in ["train", "validation", "test", "challenge"]
        },
        "rows": held,
        "methods": methods,
        "model_label_mapping": mapping,
        "nli_entailment_id": entailment,
        "selection": {
            "tfidf": baseline_candidates,
            "embedding": embedding_candidates,
            "thresholds": thresholds,
            "review_threshold": selected_threshold,
            "constraint_met": bool(feasible),
        },
        "wording": {
            "original": descriptions,
            "alternate": [
                "The viewer disliked this film.",
                "The viewer liked this film.",
            ],
            "changed_ids": wording_changes,
            "alternate_cases": [
                dict(id=r["id"], **p) for r, p in zip(held, wording_predictions)
            ],
        },
        "t5_prompt": prompt,
        "reload_checked": True,
        "shared_test_ids_checked": all(
            [r["id"] for r in m["cases"]] == [r["id"] for r in held]
            for m in methods.values()
        ),
        "truncation": "All transformer inputs cap at 256 tokens; prompts/hypotheses consume part of that budget. Raw sentence text remains visible.",
        "limitations": [
            "80 balanced test sentences; substantial sampling uncertainty.",
            "Pretrained and instruction-training overlap with review sources is unknown.",
            "Challenge gold labels are author judgments, especially on mixed sentiment.",
            "Different methods have different prior training; this is a workflow comparison, not an architecture-controlled benchmark.",
            "Timing uses one CPU thread, batch 16, three warmed passes; model loading excluded.",
        ],
    }
    save_result("e4-classification", result)
    print(
        json.dumps(
            {
                "reload_checked": True,
                "shared_test_ids_checked": result["shared_test_ids_checked"],
                "test_count": 80,
                "method_count": len(methods),
                "macro_f1": {
                    name: row["test"]["macro_f1"] for name, row in methods.items()
                },
            }
        )
    )


if __name__ == "__main__":
    main()
