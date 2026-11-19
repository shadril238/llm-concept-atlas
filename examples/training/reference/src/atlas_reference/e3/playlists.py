"""L04: learn song vectors with skip-gram negative sampling, then evaluate."""

import json
import argparse
from collections import Counter
from pathlib import Path
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[6]


def main(argv=None):
    torch.manual_seed(11)
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=ROOT / "examples/data/fixtures/playlists.json"
    )
    args = parser.parse_args(argv)
    data = json.loads(args.data.read_text())
    catalog = [row["id"] for row in data["items"]]
    index = {name: i for i, name in enumerate(catalog)}
    train = [row for row in data["playlists"] if row["split"] == "train"]
    counts = Counter(song for row in train for song in row["songs"])
    active = [index[song] for song in sorted(counts)]
    pairs = []
    for row in train:
        for i, center in enumerate(row["songs"]):
            for j in range(max(0, i - 2), min(len(row["songs"]), i + 3)):
                if i != j:
                    pairs.append((index[center], index[row["songs"][j]]))
    pairs = torch.tensor(pairs)
    center = torch.nn.Embedding(len(catalog), 8)
    context = torch.nn.Embedding(len(catalog), 8)
    torch.nn.init.normal_(center.weight, std=0.1)
    torch.nn.init.zeros_(context.weight)
    optimizer = torch.optim.Adam(
        list(center.parameters()) + list(context.parameters()), lr=0.035
    )
    noise = torch.tensor(
        [counts[catalog[i]] ** 0.75 for i in range(len(catalog))], dtype=torch.float
    )
    original = center.weight.detach().clone()
    losses = []
    for step in range(180):
        optimizer.zero_grad()
        batch = pairs[torch.randint(len(pairs), (64,))]
        u = center(batch[:, 0])
        positive = context(batch[:, 1])
        negatives = torch.multinomial(noise, 64 * 4, replacement=True).reshape(64, 4)
        negative_vectors = context(negatives)
        objective = (
            F.softplus(-(u * positive).sum(-1)).mean()
            + F.softplus((u[:, None, :] * negative_vectors).sum(-1)).sum(-1).mean()
        )
        objective.backward()
        optimizer.step()
        if step % 30 == 0:
            losses.append(float(objective.detach()))
    assert not torch.equal(original, center.weight.detach())
    vectors = F.normalize(center.weight.detach(), dim=1)

    def evaluate(split):
        cases = []
        for row in data["playlists"]:
            if row["split"] != split:
                continue
            observed, target = row["songs"][:-1], row["songs"][-1]
            if target not in counts or any(song not in counts for song in observed):
                continue
            candidates = [song for song in counts if song not in observed]
            query = F.normalize(vectors[[index[s] for s in observed]].mean(0), dim=0)
            learned = sorted(
                candidates, key=lambda s: (-float(vectors[index[s]] @ query), s)
            )
            popular = sorted(candidates, key=lambda s: (-counts[s], s))
            cases.append(
                {
                    "id": row["id"],
                    "observed": observed,
                    "target": target,
                    "learned": learned[:3],
                    "popular": popular[:3],
                    "rank": learned.index(target) + 1,
                    "popular_rank": popular.index(target) + 1,
                }
            )
        if not cases:
            raise ValueError(
                "No warm-item evaluation queries; inspect user splits and coverage"
            )
        return {
            "queries": len(cases),
            "total_queries": sum(row["split"] == split for row in data["playlists"]),
            "cold_skipped": sum(row["split"] == split for row in data["playlists"])
            - len(cases),
            "recall_at_3": sum(r["target"] in r["learned"] for r in cases) / len(cases),
            "mrr": sum(1 / r["rank"] for r in cases) / len(cases),
            "popularity_recall_at_3": sum(r["target"] in r["popular"] for r in cases)
            / len(cases),
            "popularity_mrr": sum(1 / r["popular_rank"] for r in cases) / len(cases),
            "cases": cases,
        }

    assert "cold" not in counts
    for split in ["validation", "test"]:
        if not any(r["split"] == split for r in data["playlists"]):
            raise ValueError("Dataset needs train/validation/test sessions")
    path = ROOT / "examples/artifacts/e3-playlists"
    path.mkdir(parents=True, exist_ok=True)
    torch.save({"vectors": vectors, "catalog": catalog}, path / "vectors.pt")
    reloaded = torch.load(path / "vectors.pt", weights_only=True)
    assert torch.equal(reloaded["vectors"], vectors)
    result = {
        "validation": evaluate("validation"),
        "test": evaluate("test"),
        "losses": losses,
        "trained_pairs": len(pairs),
        "item_count": len(active),
        "cold_start": "cold has no training interactions; do not recommend its random row",
        "vectors": vectors[active].tolist(),
        "catalog": [catalog[i] for i in active],
        "learned_vectors": True,
        "reload_checked": True,
    }
    (path / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps({k: v for k, v in result.items() if k not in ["vectors", "catalog"]})
    )


if __name__ == "__main__":
    main()
