"""One-thread CPU runtime, real pretrained encoders and transparent training/evaluation."""

from pathlib import Path
import hashlib
import json
import os
import random
import sys
import time
import copy

ROOT = Path(__file__).resolve().parents[5]
PROJECT = ROOT / "examples/training/representation"
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(ROOT / "examples/foundations/src"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["ACCELERATE_USE_CPU"] = "true"
os.environ["WANDB_DISABLED"] = "true"
os.environ["HF_DATASETS_OFFLINE"] = "1"
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, models, losses
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from atlas_foundations.e7.contracts import check_splits, check_mnr_batch
from download_models import download

MODEL = "prajjwal1/bert-tiny"
TOKENIZER = "google-bert/bert-base-uncased"
META = json.loads((PROJECT / "models.json").read_text())
torch.set_num_threads(1)


def seed(value=42):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def fixture():
    data = json.loads(
        (ROOT / "examples/data/fixtures/embedding-training.json").read_text()
    )
    check_splits(data)
    return data


def sentiment():
    return json.loads((ROOT / "examples/data/fixtures/sentiment.json").read_text())


def save(name, data):
    path = ROOT / "examples/artifacts" / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "results.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def artifact_files(directory):
    return [
        {
            "path": str(p.relative_to(ROOT)),
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    ]


def state_hash(model, predicate=lambda n, p: True):
    h = hashlib.sha256()
    for n, p in model.named_parameters():
        if predicate(n, p):
            h.update(n.encode())
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def parameter_report(model, before, gradients):
    groups = {}
    for n, p in model.named_parameters():
        group = (
            "embeddings"
            if "embeddings" in n
            else "layer_0"
            if "layer.0." in n
            else "layer_1"
            if "layer.1." in n
            else "head_or_pooler"
        )
        row = groups.setdefault(
            group, dict(parameters=0, trainable=0, changed_tensors=0, gradient_norm=0.0)
        )
        row["parameters"] += p.numel()
        row["trainable"] += p.numel() if p.requires_grad else 0
        row["changed_tensors"] += int(not torch.equal(before[n], p.detach().cpu()))
        row["gradient_norm"] += gradients.get(n, 0) ** 2
    for row in groups.values():
        row["gradient_norm"] = row["gradient_norm"] ** 0.5
    return groups


def encoder(pooling="mean", path=None):
    seed()
    word = models.Transformer(
        path or download(MODEL),
        max_seq_length=64,
        model_args={"local_files_only": True},
        tokenizer_args={"local_files_only": True},
    )
    pool = models.Pooling(word.get_word_embedding_dimension(), pooling_mode=pooling)
    return SentenceTransformer(modules=[word, pool], device="cpu")


def pairs(data, split):
    by = {r["id"]: r for r in data["rows"]}
    return [
        dict(p, left=by[p["left_id"]]["text"], right=by[p["right_id"]]["text"])
        for p in data["gold_pairs"]
        if p["split"] == split
    ]


def similarity_eval(model, rows):
    a = model.encode(
        [r["left"] for r in rows], normalize_embeddings=True, show_progress_bar=False
    )
    b = model.encode(
        [r["right"] for r in rows], normalize_embeddings=True, show_progress_bar=False
    )
    scores = np.sum(a * b, axis=1)
    gold = np.array([r["label"] for r in rows])
    variable = np.std(scores) > 1e-10 and np.std(gold) > 1e-10
    return dict(
        mse=float(np.mean((scores - gold) ** 2)),
        pearson=float(pearsonr(scores, gold).statistic) if variable else None,
        spearman=float(spearmanr(scores, gold).statistic) if variable else None,
        rows=[
            dict(id=r["id"], gold=r["label"], score=float(s))
            for r, s in zip(rows, scores)
        ],
    )


def retrieval_eval(model, data, split):
    group = [r for r in data["rows"] if r["split"] == split]
    intents = sorted({r["intent"] for r in group})
    docs = [next(r for r in group if r["intent"] == i) for i in intents]
    queries = [r for r in group if r not in docs]
    q = model.encode(
        [r["text"] for r in queries], normalize_embeddings=True, show_progress_bar=False
    )
    d = model.encode(
        [r["text"] for r in docs], normalize_embeddings=True, show_progress_bar=False
    )
    matrix = q @ d.T
    rows = []
    for r, scores in zip(queries, matrix):
        order = sorted(
            range(len(docs)), key=lambda i: (-float(scores[i]), docs[i]["id"])
        )
        rank = next(
            rank for rank, i in enumerate(order, 1) if docs[i]["intent"] == r["intent"]
        )
        rows.append(
            dict(
                id=r["id"],
                text=r["text"],
                intent=r["intent"],
                rank=rank,
                ranking=[
                    dict(
                        id=docs[i]["id"],
                        text=docs[i]["text"],
                        intent=docs[i]["intent"],
                        score=float(scores[i]),
                    )
                    for i in order
                ],
            )
        )
    return dict(
        mrr=sum(1 / r["rank"] for r in rows) / len(rows),
        recall_at_1=sum(r["rank"] == 1 for r in rows) / len(rows),
        queries=len(rows),
        documents=len(docs),
        rows=rows,
    )


def cls_eval(gold, pred):
    return dict(
        accuracy=float(accuracy_score(gold, pred)),
        macro_f1=float(f1_score(gold, pred, average="macro", zero_division=0)),
        confusion=confusion_matrix(gold, pred, labels=[0, 1]).tolist(),
        per_class=classification_report(
            gold, pred, labels=[0, 1], output_dict=True, zero_division=0
        ),
    )


def train_embedding(
    model,
    objective,
    data,
    train_rows,
    steps=60,
    directory=None,
    kind="cosine",
    rate=3e-4,
):
    seed()
    initial = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    optimizer = torch.optim.AdamW(objective.parameters(), lr=rate, weight_decay=0.01)
    objective_head_initial = (
        state_hash(objective.classifier) if kind == "softmax" else None
    )
    history = []
    best = None
    best_score = None
    gradients = {}
    rng = random.Random(42)
    by = {r["id"]: r for r in data["rows"]}
    for step in range(steps):
        if kind == "mnr":
            batch = []
            for intent in sorted({r["intent"] for r in data["rows"]}):
                options = [
                    r for r in train_rows if by[r["left_id"]]["intent"] == intent
                ]
                batch.append(options[step % len(options)])
            check_mnr_batch(batch, by)
        else:
            batch = rng.sample(train_rows, min(8, len(train_rows)))
        model.train()
        objective.train()
        optimizer.zero_grad()
        features = [
            model.tokenize([r[key] for r in batch]) for key in ["left", "right"]
        ]
        labels = torch.tensor(
            [r["label"] for r in batch],
            dtype=torch.long if kind == "softmax" else torch.float32,
        )
        loss = objective(features, labels)
        assert torch.isfinite(loss)
        loss.backward()
        if step == 0:
            gradients = {
                n: float(p.grad.norm())
                for n, p in model.named_parameters()
                if p.grad is not None
            }
        torch.nn.utils.clip_grad_norm_(objective.parameters(), 1)
        optimizer.step()
        if (step + 1) % 10 == 0 or step == 0:
            model.eval()
            sim = similarity_eval(model, pairs(data, "validation"))
            retrieval = retrieval_eval(model, data, "validation")
            if kind == "softmax":
                objective.eval()
                v = pairs(data, "validation")
                with torch.no_grad():
                    value = float(
                        objective(
                            [
                                model.tokenize([r[k] for r in v])
                                for k in ["left", "right"]
                            ],
                            torch.tensor([r["label"] for r in v], dtype=torch.long),
                        )
                    )
            else:
                value = -retrieval["mrr"] if kind == "mnr" else sim["mse"]
            history.append(
                dict(
                    step=step + 1,
                    training_loss=float(loss.detach()),
                    validation_mse=sim["mse"],
                    validation_mrr=retrieval["mrr"],
                    selection_value=value,
                )
            )
            if best_score is None or value < best_score:
                best_score = value
                best = copy.deepcopy(objective.state_dict())
                best_step = step + 1
    objective.load_state_dict(best)
    if kind == "softmax":
        assert objective_head_initial != state_hash(objective.classifier)
    model.eval()
    report = parameter_report(model, initial, gradients)
    assert any(r["changed_tensors"] for r in report.values())
    assert any(v > 0 for v in gradients.values())
    if directory:
        model.save_pretrained(str(directory))
        objective_path = directory / "objective.pt"
        torch.save(objective.state_dict(), objective_path)
        reloaded = SentenceTransformer(str(directory), device="cpu")
        probe = [r["text"] for r in data["rows"] if r["split"] == "validation"]
        assert np.allclose(model.encode(probe), reloaded.encode(probe), atol=1e-6)
    return dict(
        history=history,
        selected_step=best_step,
        selection="minimum validation pair cross-entropy"
        if kind == "softmax"
        else "maximum validation retrieval MRR"
        if kind == "mnr"
        else "minimum validation pair MSE",
        parameters=report,
        gradient_checked=True,
        updated=True,
        reload_verified=directory is not None,
        optimizer=dict(
            name="AdamW",
            learning_rate=rate,
            weight_decay=0.01,
            gradient_clip=1,
            batch_size=6 if kind == "mnr" else 8,
            steps=steps,
            seed=42,
        ),
    )


def classifier_run(
    train, valid, test, directory, freeze="full", backbone=None, run_seed=42, steps=60
):
    seed(run_seed)
    tokenizer = AutoTokenizer.from_pretrained(download(MODEL), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        download(MODEL), num_labels=2, local_files_only=True
    ).cpu()
    # MLM checkpoints omit BERT's pooler. Preserve the identical original pooler/head
    # and replace only the shared embeddings/encoder, so initialization is controlled.
    if backbone:
        adapted = AutoModel.from_pretrained(
            backbone, add_pooling_layer=False, local_files_only=True
        ).cpu()
        incompat = model.bert.load_state_dict(adapted.state_dict(), strict=False)
        assert (
            set(incompat.missing_keys) == {"pooler.dense.weight", "pooler.dense.bias"}
            and not incompat.unexpected_keys
        )
    head_pooler_initial_sha256 = state_hash(
        model, lambda n, p: n.startswith("classifier.") or ".pooler." in n
    )
    for name, p in model.named_parameters():
        if freeze == "head-only" and name.startswith("bert."):
            p.requires_grad = False
        elif freeze == "partial" and (
            name.startswith("bert.embeddings.")
            or name.startswith("bert.encoder.layer.0.")
        ):
            p.requires_grad = False
    before = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    frozen = state_hash(model, lambda n, p: not p.requires_grad)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=0.01
    )
    rng = random.Random(run_seed)
    history = []
    best = None
    score = None
    gradients = {}

    def predict(rows, net=model):
        net.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(rows), 16):
                batch = rows[start : start + 16]
                inputs = tokenizer(
                    [r["text"] for r in batch],
                    padding=True,
                    truncation=True,
                    max_length=96,
                    return_tensors="pt",
                )
                outputs.extend(net(**inputs).logits.softmax(-1).tolist())
        return outputs

    initial_predictions = np.argmax(predict(valid), axis=1).tolist()
    # Loading an extra adapted backbone consumes RNG even when weights are loaded.
    # Reset training randomness so the downstream comparison shares dropout draws.
    seed(run_seed)
    for step in range(steps):
        model.train()
        # Freezing a parameter does not disable dropout; explicitly freeze representation stochasticity for the head-only route.
        if freeze == "head-only":
            model.bert.eval()
        batch = rng.sample(train, min(8, len(train)))
        optimizer.zero_grad()
        inputs = tokenizer(
            [r["text"] for r in batch],
            padding=True,
            truncation=True,
            max_length=96,
            return_tensors="pt",
        )
        loss = model(**inputs, labels=torch.tensor([r["label"] for r in batch])).loss
        assert torch.isfinite(loss)
        loss.backward()
        if step == 0:
            gradients = {
                n: float(p.grad.norm())
                for n, p in model.named_parameters()
                if p.grad is not None
            }
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if (step + 1) % 10 == 0:
            probs = predict(valid)
            pred = np.argmax(probs, axis=1).tolist()
            value = cls_eval([r["label"] for r in valid], pred)["macro_f1"]
            history.append(
                dict(
                    step=step + 1,
                    training_loss=float(loss.detach()),
                    validation_macro_f1=value,
                )
            )
            if score is None or value > score:
                score = value
                best = copy.deepcopy(model.state_dict())
                best_step = step + 1
    model.load_state_dict(best)
    model.eval()
    assert frozen == state_hash(model, lambda n, p: not p.requires_grad)
    report = parameter_report(model, before, gradients)
    assert any(r["changed_tensors"] for r in report.values())
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    reloaded = AutoModelForSequenceClassification.from_pretrained(directory).eval()
    assert np.allclose(predict(valid), predict(valid, reloaded), atol=1e-6)
    probabilities = predict(test)
    pred = np.argmax(probabilities, axis=1).tolist()
    return dict(
        head_pooler_initial_sha256=head_pooler_initial_sha256,
        freeze=freeze,
        seed=run_seed,
        steps=steps,
        selected_step=best_step,
        selection="first maximum validation macro-F1 at fixed 10-step checkpoints",
        history=history,
        parameters=report,
        frozen_verified=True,
        reload_verified=True,
        initial_validation=cls_eval([r["label"] for r in valid], initial_predictions),
        test=cls_eval([r["label"] for r in test], pred),
        predictions=[
            dict(
                id=r["id"],
                text=r["text"],
                gold=r["label"],
                prediction=p,
                probabilities=probs,
            )
            for r, p, probs in zip(test, pred, probabilities)
        ],
        train_ids=[r["id"] for r in train],
        artifact=artifact_files(directory),
    )
