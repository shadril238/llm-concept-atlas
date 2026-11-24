"""L16: train a cross-encoder teacher, label training-only pairs, compare two students."""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas_representation.common import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    MODEL,
    ROOT,
    artifact_files,
    cls_eval,
    copy,
    download,
    encoder,
    fixture,
    losses,
    pairs,
    random,
    retrieval_eval,
    save,
    seed,
    similarity_eval,
    state_hash,
    torch,
    train_embedding,
)


def main():
    seed()
    data = fixture()
    gold = pairs(data, "train")
    valid = pairs(data, "validation")
    test = pairs(data, "test")
    by = {r["id"]: r for r in data["rows"]}
    tokenizer = AutoTokenizer.from_pretrained(download(MODEL), local_files_only=True)
    teacher = AutoModelForSequenceClassification.from_pretrained(
        download(MODEL), num_labels=2, local_files_only=True
    ).cpu()
    initial = state_hash(teacher)
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=3e-4, weight_decay=0.01)
    history = []
    best = None
    score = None
    gradient = None
    rng = random.Random(42)

    def predict(rows):
        teacher.eval()
        with torch.no_grad():
            return teacher(
                **tokenizer(
                    [r["left"] for r in rows],
                    [r["right"] for r in rows],
                    padding=True,
                    truncation=True,
                    max_length=64,
                    return_tensors="pt",
                )
            ).logits.softmax(-1)[:, 1]

    for step in range(60):
        batch = rng.sample(gold, 8)
        teacher.train()
        optimizer.zero_grad()
        inputs = tokenizer(
            [r["left"] for r in batch],
            [r["right"] for r in batch],
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        labels = torch.tensor([int(r["label"]) for r in batch])
        loss = teacher(**inputs, labels=labels).loss
        loss.backward()
        if step == 0:
            gradient = float(
                teacher.bert.encoder.layer[0].attention.self.query.weight.grad.norm()
            )
        torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1)
        optimizer.step()
        if (step + 1) % 10 == 0:
            probs = predict(valid)
            labels = torch.tensor([r["label"] for r in valid])
            value = float(torch.nn.functional.binary_cross_entropy(probs, labels))
            history.append(
                dict(
                    step=step + 1, loss=float(loss.detach()), validation_binary_ce=value
                )
            )
            if score is None or value < score:
                score = value
                best = copy.deepcopy(teacher.state_dict())
                best_step = step + 1
    teacher.load_state_dict(best)
    assert state_hash(teacher) != initial and gradient > 0
    folder = ROOT / "examples/artifacts/e7-augmentation/teacher"
    teacher.save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    before = predict(valid)
    reloaded = AutoModelForSequenceClassification.from_pretrained(folder).eval()
    with torch.no_grad():
        after = reloaded(
            **tokenizer(
                [r["left"] for r in valid],
                [r["right"] for r in valid],
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )
        ).logits.softmax(-1)[:, 1]
    assert torch.allclose(before, after, atol=1e-6)
    candidates = [
        dict(p, left=by[p["left_id"]]["text"], right=by[p["right_id"]]["text"])
        for p in data["silver_candidates"]
    ]
    scores = predict(candidates).tolist()
    silver = [
        dict(p, label=float(value), origin="trained teacher probability, not gold")
        for p, value in zip(candidates, scores)
    ]
    # Hidden fixture intents are used after prediction for diagnostics only, never teacher targets.
    diagnostics = [
        dict(
            id=p["id"],
            teacher_score=p["label"],
            latent_same_intent=by[p["left_id"]]["intent"]
            == by[p["right_id"]]["intent"],
            disagreement=(p["label"] >= 0.5)
            != (by[p["left_id"]]["intent"] == by[p["right_id"]]["intent"]),
        )
        for p in silver
    ]
    students = []
    base = encoder()
    baseline = similarity_eval(base, test)
    for kind, rows in [("gold-only", gold), ("gold-plus-silver", gold + silver)]:
        model = encoder()
        objective = losses.CosineSimilarityLoss(model)
        path = ROOT / f"examples/artifacts/e7-augmentation/{kind}"
        training = train_embedding(
            model, objective, data, rows, steps=60, directory=path
        )
        students.append(
            dict(
                condition=kind,
                training=training,
                similarity=similarity_eval(model, test),
                retrieval=retrieval_eval(model, data, "test"),
                train_pairs=len(rows),
                artifact=artifact_files(path),
            )
        )
    result = dict(
        teacher=dict(
            history=history,
            selected_step=best_step,
            initial_sha256=initial,
            selected_sha256=state_hash(teacher),
            first_query_gradient_norm=gradient,
            validation_scores=before.tolist(),
            test=cls_eval(
                [int(r["label"]) for r in test], [int(x >= 0.5) for x in predict(test)]
            ),
            artifact=artifact_files(folder),
        ),
        baseline=baseline,
        students=students,
        silver=silver,
        diagnostics=diagnostics,
        protocol=data["protocol"],
        comparison_limit="Equal 60-step student budgets and initialization; adding silver changes pair distribution and exposure, not only label quality. Teacher errors remain visible; no independent benchmark claim.",
        verified=True,
    )
    save("e7-augmentation", result)
    print(
        json.dumps(
            dict(
                teacher_trained=True,
                silver_split_verified=True,
                students_updated=True,
                reload_verified=True,
                silver_pairs=len(silver),
            )
        )
    )


if __name__ == "__main__":
    main()
