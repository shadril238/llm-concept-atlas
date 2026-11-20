"""L08: fitted BERTopic, controlled representation/density changes, safe reload."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas_discovery.common import ROOT, Encoder, save_result, setup
from atlas_foundations.e4.topic_math import class_tfidf

import numpy as np
from bertopic import BERTopic
from bertopic.backend import BaseEmbedder
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP


class PinnedBackend(BaseEmbedder):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()

    def embed(self, documents, verbose=False):
        return self.encoder.encode(documents)


def terms(model):
    return {
        str(topic): [
            {"term": word, "weight": float(weight)} for word, weight in values[:5]
        ]
        for topic, values in model.get_topics().items()
    }


# region pipeline
def fit_pipeline(documents, embeddings, backend):
    """Documents -> real UMAP -> real HDBSCAN -> class-based TF-IDF."""
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    reducer = UMAP(
        n_neighbors=8,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
        n_jobs=1,
    )
    clusterer = HDBSCAN(
        min_cluster_size=5,
        min_samples=8,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 1))
    model = BERTopic(
        embedding_model=backend,
        umap_model=reducer,
        hdbscan_model=clusterer,
        vectorizer_model=vectorizer,
        ctfidf_model=ClassTfidfTransformer(),
        top_n_words=5,
        calculate_probabilities=False,
    )
    assignments, strengths = model.fit_transform(documents, embeddings)
    return model, assignments, strengths


# endregion pipeline


def main():
    setup()
    corpus = json.loads(
        (ROOT / "examples/data/fixtures/research-summaries.json").read_text()
    )
    rows = corpus["documents"]
    documents = [row["text"] for row in rows]
    backend = PinnedBackend()
    embeddings = backend.embed(documents)
    start = time.perf_counter()
    model, assignments, probabilities = fit_pipeline(documents, embeddings, backend)
    fit_seconds = time.perf_counter() - start
    assignments = np.asarray(assignments)
    assert len(assignments) == len(documents)
    assert len(set(assignments) - {-1}) >= 2, (
        "Inspect corpus/settings; fewer than two fitted topics"
    )
    reduced = np.asarray(model.umap_model.embedding_)
    vectorizer = model.vectorizer_model
    display = UMAP(
        n_neighbors=8,
        n_components=2,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
        n_jobs=1,
    ).fit_transform(embeddings)
    baseline = terms(model)
    representative = model.get_representative_docs()
    topic_ids = sorted(int(topic) for topic in baseline)
    topic_embeddings = np.array(model.topic_embeddings_)
    similarities = cosine_similarity(topic_embeddings)
    non_outlier = [i for i, topic in enumerate(topic_ids) if topic != -1]
    # An inspectable hierarchy uses average linkage on cosine distances between
    # the original fitted topic vectors. It does not change assignments.
    distance = np.clip(1 - similarities[np.ix_(non_outlier, non_outlier)], 0, 2)
    np.fill_diagonal(distance, 0)
    hierarchy = linkage(
        squareform((distance + distance.T) / 2, checks=True), method="average"
    ).tolist()
    out = ROOT / "examples/artifacts/e4-topics"
    out.mkdir(parents=True, exist_ok=True)
    model.save(
        str(out / "model"),
        serialization="safetensors",
        save_ctfidf=True,
        save_embedding_model=False,
    )
    restored = BERTopic.load(str(out / "model"), embedding_model=backend)
    assert np.array_equal(np.asarray(restored.topics_), assignments)
    for topic in topic_ids:
        assert [w for w, _ in restored.get_topic(topic)] == [
            w for w, _ in model.get_topic(topic)
        ]
    assert np.allclose(
        restored.c_tf_idf_.toarray(), model.c_tf_idf_.toarray(), atol=1e-7
    )
    restored_predictions, _ = restored.transform(documents, embeddings=embeddings)
    expected = cosine_similarity(embeddings, restored.topic_embeddings_).argmax(
        1
    ) - int(-1 in topic_ids)
    assert np.array_equal(restored_predictions, expected)
    baseline_reloaded_agreement = float(np.mean(assignments == restored_predictions))

    # Both membership and geometry are frozen for each representation-only pass.
    representations = {"ctfidf": baseline}
    for name, representation in [
        (
            "keybert",
            KeyBERTInspired(
                top_n_words=5,
                nr_repr_docs=3,
                nr_samples=100,
                nr_candidate_words=30,
                random_state=42,
            ),
        ),
        ("mmr-0.2", MaximalMarginalRelevance(diversity=0.2, top_n_words=5)),
        ("mmr-0.8", MaximalMarginalRelevance(diversity=0.8, top_n_words=5)),
    ]:
        model.update_topics(
            documents,
            topics=assignments.tolist(),
            vectorizer_model=vectorizer,
            ctfidf_model=ClassTfidfTransformer(),
            representation_model=representation,
            top_n_words=5,
        )
        assert np.array_equal(np.asarray(model.topics_), assignments)
        representations[name] = terms(model)

    # Change one density control on the SAME five-dimensional points; compare
    # label-invariant ARI and membership/noise transitions, not raw ID equality.
    changed = HDBSCAN(
        min_cluster_size=5,
        min_samples=3,
        metric="euclidean",
        cluster_selection_method="eom",
    ).fit_predict(reduced)
    contingency = [
        {
            "before": int(a),
            "after": int(b),
            "count": int(np.sum((assignments == a) & (changed == b))),
        }
        for a in sorted(set(assignments))
        for b in sorted(set(changed))
        if np.any((assignments == a) & (changed == b))
    ]
    arithmetic_counts = [[3, 1, 0], [0, 1, 3]]
    arithmetic = class_tfidf(arithmetic_counts)
    actual = (
        ClassTfidfTransformer().fit_transform(csr_matrix(arithmetic_counts)).toarray()
    )
    assert np.allclose(actual, arithmetic["weights"], atol=1e-12)
    result = {
        "corpus": corpus["id"],
        "fit_seconds": fit_seconds,
        "shapes": {
            "embeddings": list(embeddings.shape),
            "clustering_space": list(reduced.shape),
            "display_space": list(display.shape),
            "ctfidf": list(restored.c_tf_idf_.shape),
        },
        "settings": {
            "umap_neighbors": 8,
            "cluster_dimensions": 5,
            "display_dimensions": 2,
            "umap_seed": 42,
            "hdbscan_min_cluster_size": 5,
            "hdbscan_min_samples": 8,
            "changed_min_samples": 3,
            "stop_words": "scikit-learn English",
            "top_n_words": 5,
        },
        "documents": [
            dict(
                row,
                topic=int(topic),
                changed_topic=int(other),
                membership_strength=float(strength),
                cluster_coordinates=point.tolist(),
                display_coordinates=xy.tolist(),
            )
            for row, topic, other, strength, point, xy in zip(
                rows, assignments, changed, probabilities, reduced, display
            )
        ],
        "topics": [
            {
                "id": topic,
                "count": int(np.sum(assignments == topic)),
                "representative_ids": [
                    row["id"]
                    for row in rows
                    if row["text"] in representative.get(topic, [])
                ],
            }
            for topic in topic_ids
        ],
        "representations": representations,
        "topic_similarity": {"topic_ids": topic_ids, "cosines": similarities.tolist()},
        "hierarchy": {
            "topic_ids": [topic_ids[i] for i in non_outlier],
            "linkage": hierarchy,
            "method": "average linkage of cosine distances between baseline topic embeddings; excludes -1",
        },
        "density_comparison": {
            "ari_including_noise": float(adjusted_rand_score(assignments, changed)),
            "before_noise": int(np.sum(assignments == -1)),
            "after_noise": int(np.sum(changed == -1)),
            "contingency": contingency,
            "new_noise_ids": [
                row["id"]
                for row, a, b in zip(rows, assignments, changed)
                if a != -1 and b == -1
            ],
        },
        "ctfidf_arithmetic": dict(
            arithmetic,
            counts=arithmetic_counts,
            vocabulary=["translation", "model", "speech"],
            trusted_library_match=True,
        ),
        "representation_assignments_unchanged": True,
        "reload_checked": True,
        "reload_inference": {
            "method": "cosine to saved topic embeddings; lightweight safetensors does not retain the fitted UMAP/HDBSCAN estimators",
            "agreement_with_original_training_assignments": baseline_reloaded_agreement,
            "reference_calculation_checked": True,
        },
        "limitations": [
            "Original fictional summaries, not actual arXiv findings; curated themes are intentionally visible.",
            "All documents participate in exploratory fitting; no held-out topic-accuracy claim.",
            "Membership strength is a density-model quantity, not the probability that a semantic topic is correct.",
            "The independent 2D UMAP display is not the 5D clustering space; apparent separation can mislead.",
            "Topic IDs are run-local. ARI compares partitions without assuming topic ID equality; -1 is treated as a group in that summary.",
            "A coherent label cannot establish topic stability or factual support.",
        ],
    }
    save_result("e4-topics", result)
    print(
        json.dumps(
            {
                "document_count": len(rows),
                "topic_count": len(non_outlier),
                "outliers": int(np.sum(assignments == -1)),
                "representation_assignments_unchanged": True,
                "reload_checked": True,
                "ctfidf_library_match": True,
                "density_ari": result["density_comparison"]["ari_including_noise"],
            }
        )
    )


if __name__ == "__main__":
    main()
