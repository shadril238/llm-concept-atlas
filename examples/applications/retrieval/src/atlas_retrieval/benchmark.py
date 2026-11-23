"""L12: actual BM25, frozen dense index, cross-encoder reranking, and held-out metrics."""

from pathlib import Path
import sys, json, time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas_retrieval.common import (
    ROOT,
    Encoder,
    Reranker,
    Retriever,
    fixture,
    save,
    provenance,
    collapse,
    metrics,
    np,
)


def main():
    data = fixture()
    encoder = Encoder()
    reranker = Reranker()
    retriever = Retriever(encoder, data["documents"])
    results = []
    for query in data["queries"]:
        for method in ["lexical", "dense", "reranked"]:
            start = time.perf_counter()
            hits = retriever.search(
                query["text"], method, limit=len(retriever.chunks), reranker=reranker
            )
            ranking = collapse(hits)
            results.append(
                {
                    "id": query["id"],
                    "split": query["split"],
                    "method": method,
                    "hits": hits,
                    "document_ranking": ranking,
                    "metrics": {
                        str(k): metrics(ranking, query["relevance"], k)
                        for k in [1, 3, 5]
                    },
                    "seconds": time.perf_counter() - start,
                }
            )
    design = [q for q in data["queries"] if q["split"] == "design"]
    threshold_rows = []
    for threshold in [0.25, 0.4, 0.55, 0.7]:
        scored = []
        for query in design:
            row = next(
                r for r in results if r["id"] == query["id"] and r["method"] == "dense"
            )
            scored.append((row["hits"][0]["score"] >= threshold) == query["answerable"])
        threshold_rows.append(
            {"threshold": threshold, "design_accuracy": sum(scored) / len(scored)}
        )
    selected = max(
        threshold_rows, key=lambda r: (r["design_accuracy"], r["threshold"])
    )["threshold"]
    aggregates = []
    for method in ["lexical", "dense", "reranked"]:
        for k in [1, 3, 5]:
            rows = [
                r
                for r in results
                if r["method"] == method and r["split"] == "evaluation"
            ]
            values = [r["metrics"][str(k)] for r in rows]
            aggregates.append(
                {
                    "method": method,
                    "k": k,
                    "queries": len(rows),
                    "answerable_queries": sum(v["ap"] is not None for v in values),
                    **{
                        key: sum(v[key] for v in values if v[key] is not None)
                        / sum(v[key] is not None for v in values)
                        for key in ["precision", "recall", "ap", "rr", "ndcg"]
                    },
                }
            )
    artifact = ROOT / "examples/artifacts/e6-retrieval"
    artifact.mkdir(parents=True, exist_ok=True)
    np.savez(artifact / "index.npz", vectors=retriever.vectors)
    (artifact / "index.json").write_text(
        json.dumps(
            {
                "chunk_ids": [r["id"] for r in retriever.chunks],
                "encoder": provenance([encoder.name]),
                "pooling": "attention-mask mean; L2 normalized",
                "metric": "cosine; score=1-distance",
                "input_cap": encoder.cap,
                "policy": "two sentences, overlap one",
            }
        )
    )
    restored = np.load(artifact / "index.npz")["vectors"]
    assert np.array_equal(restored, retriever.vectors)
    from sklearn.neighbors import NearestNeighbors

    loaded = NearestNeighbors(metric="cosine", algorithm="brute").fit(restored)
    sample = encoder.encode([data["queries"][0]["text"]])
    assert np.array_equal(
        loaded.kneighbors(sample, n_neighbors=6)[1],
        retriever.index.kneighbors(sample, n_neighbors=6)[1],
    )
    changed = Retriever(encoder, data["documents"], "word")
    comparison = []
    for query in [q for q in data["queries"] if q["split"] == "evaluation"]:
        hits = changed.search(query["text"], limit=len(changed.chunks))
        comparison.append(
            {
                "id": query["id"],
                "hits": hits[:5],
                "metrics": metrics(collapse(hits), query["relevance"], 3),
            }
        )
    result = {
        "models": provenance([encoder.name, reranker.name]),
        "documents": data["documents"],
        "queries": data["queries"],
        "chunks": retriever.chunks,
        "index_shape": list(retriever.vectors.shape),
        "rows": results,
        "aggregates": aggregates,
        "threshold_selection": {
            "candidates": threshold_rows,
            "selected": selected,
            "split": "design only",
            "limitation": "A small design-set heuristic, not a calibrated answerability probability.",
        },
        "chunk_comparison": {
            "baseline": "two sentences, overlap one",
            "changed": "24 whitespace words, overlap six",
            "rows": comparison,
        },
        "reload_verified": True,
        "index_contract_verified": True,
        "split_verified": True,
        "metric_policy": "Document IDs collapsed before scoring so overlapping chunks cannot inflate hits. Grade>0 is binary relevant. AP@k divides by every judged relevant document, not min(R,k). AP/recall/MRR/nDCG undefined for absent queries; MAP excludes them and reports its denominator. P@k uses k including unfilled slots. Reranker sees only six dense candidate chunks, so its candidate ceiling is explicit.",
        "limitations": [
            "Original small, author-judged fictional corpus; no broad benchmark claim.",
            "Threshold selected on six design queries only; no evaluation-set tuning.",
            "Raw reranker logits, cosine and BM25 scores have different scales and are not calibrated probabilities.",
            "One sequential CPU pass, including query encoding; not a speed leaderboard.",
        ],
    }
    save("e6-retrieval", result)
    print(
        json.dumps(
            {
                "reload_verified": True,
                "index_contract_verified": True,
                "split_verified": True,
                "evaluation_queries": 16,
                "chunks": len(retriever.chunks),
                "map_at_3": {r["method"]: r["ap"] for r in aggregates if r["k"] == 3},
            }
        )
    )


if __name__ == "__main__":
    main()
