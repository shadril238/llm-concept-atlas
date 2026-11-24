"""L15: real sentence-transformer losses, parameter updates and held-out embedding quality."""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas_representation.common import (
    META,
    MODEL,
    ROOT,
    artifact_files,
    cls_eval,
    encoder,
    fixture,
    losses,
    pairs,
    retrieval_eval,
    save,
    seed,
    similarity_eval,
    torch,
    train_embedding,
)
from atlas_foundations.e7.contracts import cross_entropy, mnr_loss
from sklearn.decomposition import PCA


def main():
    seed()
    data = fixture()
    gold = pairs(data, "train")
    val = pairs(data, "validation")
    test = pairs(data, "test")
    baseline = encoder()
    base = dict(
        similarity=similarity_eval(baseline, test),
        retrieval=retrieval_eval(baseline, data, "test"),
    )
    view_rows = [r for r in data["rows"] if r["split"] == "test"]
    fit_texts = [r["text"] for r in data["rows"] if r["split"] == "train"]
    pca = PCA(n_components=2).fit(baseline.encode(fit_texts, normalize_embeddings=True))
    base_points = pca.transform(
        baseline.encode([r["text"] for r in view_rows], normalize_embeddings=True)
    )
    results = []
    checks = {}
    for kind in ["softmax", "cosine", "mnr"]:
        model = encoder()
        objective = (
            losses.SoftmaxLoss(model, 128, 2)
            if kind == "softmax"
            else losses.CosineSimilarityLoss(model)
            if kind == "cosine"
            else losses.MultipleNegativesRankingLoss(model, scale=20)
        )
        selected = gold if kind != "mnr" else [r for r in gold if r["label"] == 1]
        probe = (
            selected[:4]
            if kind != "mnr"
            else [
                next(r for r in selected if r["left_id"].split("-")[1] == i)
                for i in sorted({r["intent"] for r in data["rows"]})
            ]
        )
        model.eval()
        objective.eval()
        features = [model.tokenize([r[k] for r in probe]) for k in ["left", "right"]]
        with torch.no_grad():
            if kind == "softmax":
                reps, logits = objective(features, None)
                target = int(probe[0]["label"])
                checks[kind] = dict(
                    concatenated_width=3 * 128,
                    logits=logits[0].tolist(),
                    target=target,
                    manual=cross_entropy(logits[0].tolist(), target),
                    library=float(
                        torch.nn.functional.cross_entropy(
                            logits[:1], torch.tensor([target])
                        )
                    ),
                )
            elif kind == "cosine":
                u = model(features[0])["sentence_embedding"]
                v = model(features[1])["sentence_embedding"]
                cos = torch.nn.functional.cosine_similarity(u, v)
                targets = torch.tensor([r["label"] for r in probe])
                checks[kind] = dict(
                    scores=cos.tolist(),
                    targets=targets.tolist(),
                    manual=float(((cos - targets) ** 2).mean()),
                    library=float(
                        objective(
                            [
                                model.tokenize([r[k] for r in probe])
                                for k in ["left", "right"]
                            ],
                            targets,
                        )
                    ),
                )
            else:
                u = torch.nn.functional.normalize(
                    model(features[0])["sentence_embedding"], dim=-1
                )
                v = torch.nn.functional.normalize(
                    model(features[1])["sentence_embedding"], dim=-1
                )
                scores = (u @ v.T).tolist()
                checks[kind] = dict(
                    scores=scores,
                    scale=20,
                    manual=mnr_loss(scores)["loss"],
                    library=float(
                        torch.nn.functional.cross_entropy(
                            torch.tensor(scores) * 20, torch.arange(len(scores))
                        )
                    ),
                )
        assert abs(checks[kind]["manual"] - checks[kind]["library"]) < 2e-5
        directory = ROOT / f"examples/artifacts/e7-embedding/{kind}"
        training = train_embedding(
            model, objective, data, selected, directory=directory, kind=kind
        )
        result = dict(
            kind=kind,
            training=training,
            similarity=similarity_eval(model, test),
            retrieval=retrieval_eval(model, data, "test"),
            artifact=artifact_files(directory),
        )
        if kind == "softmax":
            objective.eval()
            with torch.no_grad():
                reps, logits = objective(
                    [model.tokenize([r[k] for r in test]) for k in ["left", "right"]],
                    None,
                )
            result["pair_classification"] = cls_eval(
                [int(r["label"]) for r in test], logits.argmax(-1).tolist()
            )
        points = pca.transform(
            model.encode([r["text"] for r in view_rows], normalize_embeddings=True)
        )
        result["projection"] = [
            dict(
                id=r["id"],
                text=r["text"],
                intent=r["intent"],
                before=a.tolist(),
                after=b.tolist(),
            )
            for r, a, b in zip(view_rows, base_points, points)
        ]
        results.append(result)
    save(
        "e7-embedding",
        dict(
            model=META[MODEL],
            baseline=base,
            conditions=results,
            loss_checks=checks,
            projection_policy="One PCA fitted to baseline training vectors, then frozen for test before/after projections. Two displayed dimensions omit information; full 128-D similarities drive evaluation.",
            protocol=data["protocol"],
            verified=True,
        ),
    )
    print(
        json.dumps(
            dict(
                training_updates_verified=True,
                loss_arithmetic_verified=True,
                reload_verified=True,
                conditions=len(results),
            )
        )
    )


if __name__ == "__main__":
    main()
