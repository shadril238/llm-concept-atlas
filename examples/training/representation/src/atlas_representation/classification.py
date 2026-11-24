"""L18: matched few-shot selection, explicit freezing and genuine SetFit training."""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas_representation.common import (
    ROOT,
    SentenceTransformer,
    accuracy_score,
    artifact_files,
    classifier_run,
    cls_eval,
    download,
    np,
    random,
    save,
    seed,
    sentiment,
    state_hash,
)
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from setfit import SetFitModel
from setfit.sampler import ContrastiveDataset
from sentence_transformers import losses
import torch

SENTENCE = "sentence-transformers/paraphrase-MiniLM-L3-v2"


def main():
    source = sentiment()["rows"]
    train = [r for r in source if r["split"] == "train"]
    valid = [r for r in source if r["split"] == "validation"]
    test = [r for r in source if r["split"] == "test"]
    runs = []
    labels = [r["label"] for r in test]
    for run_seed in [11, 23, 37]:
        rng = random.Random(run_seed)
        sample = []
        for label in [0, 1]:
            sample += rng.sample([r for r in train if r["label"] == label], 8)
        for policy in ["full", "partial", "head-only"]:
            directory = (
                ROOT / f"examples/artifacts/e7-classification/seed-{run_seed}-{policy}"
            )
            r = classifier_run(
                sample, valid, test, directory, freeze=policy, run_seed=run_seed
            )
            r["method"] = "bert-" + policy
            runs.append(r)
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        x = vectorizer.fit_transform([r["text"] for r in sample])
        clf = LogisticRegression(random_state=run_seed, max_iter=1000).fit(
            x, [r["label"] for r in sample]
        )
        pred = clf.predict(vectorizer.transform([r["text"] for r in test])).tolist()
        runs.append(
            dict(
                method="tfidf-few-shot",
                seed=run_seed,
                train_ids=[r["id"] for r in sample],
                test=cls_eval(labels, pred),
                predictions=[
                    dict(id=r["id"], gold=r["label"], prediction=p)
                    for r, p in zip(test, pred)
                ],
            )
        )
        seed(run_seed)
        body = SentenceTransformer(download(SENTENCE), device="cpu")
        body.max_seq_length = 96
        initial = state_hash(body)
        probe = body.encode(
            [r["text"] for r in sample],
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        head = LogisticRegression(random_state=run_seed, max_iter=1000).fit(
            probe, [r["label"] for r in sample]
        )
        pred = head.predict(
            body.encode(
                [r["text"] for r in test],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        ).tolist()
        runs.append(
            dict(
                method="frozen-sentence-logistic",
                seed=run_seed,
                train_ids=[r["id"] for r in sample],
                test=cls_eval(labels, pred),
                predictions=[
                    dict(id=r["id"], gold=r["label"], prediction=p)
                    for r, p in zip(test, pred)
                ],
            )
        )
        model = SetFitModel(
            model_body=body,
            model_head=LogisticRegression(random_state=run_seed, max_iter=1000),
            normalize_embeddings=True,
        )
        gradient = []
        handle = (
            body[0]
            .auto_model.encoder.layer[0]
            .attention.self.query.weight.register_hook(
                lambda g: gradient.append(float(g.norm())) if not gradient else None
            )
        )
        # Use the real SetFit sampler/model with an explicit CPU optimizer loop.
        # SetFit 1.1.3's convenience Trainer implicitly selects MPS on this host;
        # the transparent two-stage path avoids claiming that as a CPU run.
        sampled_pairs = list(
            ContrastiveDataset(
                [r["text"] for r in sample],
                [r["label"] for r in sample],
                multilabel=False,
                num_iterations=4,
            )
        )
        objective = losses.CosineSimilarityLoss(body)
        optimizer = torch.optim.AdamW(body.parameters(), lr=2e-4, weight_decay=0.01)
        history = []
        pair_rng = random.Random(run_seed)
        for step in range(24):
            batch = pair_rng.sample(sampled_pairs, 8)
            body.train()
            optimizer.zero_grad()
            features = [
                body.tokenize([r[k] for r in batch])
                for k in ["sentence_1", "sentence_2"]
            ]
            loss = objective(features, torch.tensor([r["label"] for r in batch]))
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(body.parameters(), 1)
            optimizer.step()
            if (step + 1) % 4 == 0 or step == 0:
                history.append(dict(step=step + 1, embedding_loss=float(loss.detach())))
        model.fit(
            [r["text"] for r in sample],
            [r["label"] for r in sample],
            num_epochs=1,
            batch_size=8,
            show_progress_bar=False,
        )
        handle.remove()
        updated = state_hash(body)
        assert updated != initial and gradient[0] > 0
        assert all(p.device.type == "cpu" for p in body.parameters())
        pred = np.asarray(model.predict([r["text"] for r in test])).tolist()
        path = ROOT / f"examples/artifacts/e7-classification/setfit-{run_seed}"
        model.save_pretrained(str(path))
        reloaded = SetFitModel.from_pretrained(
            str(path), local_files_only=True, device="cpu"
        )
        assert pred == np.asarray(reloaded.predict([r["text"] for r in test])).tolist()
        runs.append(
            dict(
                method="setfit",
                seed=run_seed,
                train_ids=[r["id"] for r in sample],
                test=cls_eval(labels, pred),
                predictions=[
                    dict(id=r["id"], text=r["text"], gold=r["label"], prediction=p)
                    for r, p in zip(test, pred)
                ],
                initial_sha256=initial,
                updated_sha256=updated,
                first_query_gradient_norm=gradient[0],
                history=history,
                settings=dict(
                    actual_training_device=str(next(body.parameters()).device),
                    generated_pair_rows=len(sampled_pairs),
                    distinct_pair_rows=len(
                        {(r["sentence_1"], r["sentence_2"]) for r in sampled_pairs}
                    ),
                    max_steps=24,
                    batch_size=8,
                    num_iterations=4,
                    body_learning_rate=2e-4,
                    head="sklearn LogisticRegression",
                    normalize_embeddings=True,
                ),
                reload_verified=True,
                artifact=artifact_files(path),
            )
        )
    summary = []
    for method in sorted({r["method"] for r in runs}):
        scores = [r["test"]["macro_f1"] for r in runs if r["method"] == method]
        summary.append(
            dict(
                method=method,
                mean_macro_f1=float(np.mean(scores)),
                std_macro_f1=float(np.std(scores)),
                minimum=min(scores),
                maximum=max(scores),
                seeds=[11, 23, 37],
            )
        )
    previous = json.loads((ROOT / "src/data/discovery-results.json").read_text())[
        "classification"
    ]["methods"]["tfidf-logistic"]["test"]
    save(
        "e7-classification",
        dict(
            runs=runs,
            summary=summary,
            earlier_full_data_tfidf=dict(
                metrics=previous,
                source="E4 recorded tfidf-logistic, same test IDs, 200 training examples instead of 16; descriptive reference, not matched-label comparison",
            ),
            protocol="Three fixed seeds; eight training rows per class, identical selected IDs across methods within each seed. Validation selects BERT checkpoints only; SetFit fixed 24 steps, never test-selected. SetFit/frozen sentence route use pretrained MiniLM sentence embeddings, whereas BERT routes use tiny BERT: architecture and prior training differ. Standard deviation over three selections is descriptive, not a confidence interval.",
            verified=True,
        ),
    )
    print(
        json.dumps(
            dict(
                frozen_parameters_verified=True,
                setfit_trained=True,
                selection_repeated=True,
                reload_verified=True,
                seeds=3,
            )
        )
    )


if __name__ == "__main__":
    main()
