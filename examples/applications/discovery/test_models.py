"""Independent library/artifact contracts; requires the completed E4 local runs."""

import json
from pathlib import Path
import sys
import unittest

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "src"))
from atlas_discovery.common import ROOT, Encoder, setup
from atlas_discovery.topics import PinnedBackend
from atlas_foundations.e4.metrics import classification_metrics
from atlas_foundations.e4.topic_math import class_tfidf
import numpy as np
import torch
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score


class RealLibraryContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup()

    def test_ctfidf_unequal_class_lengths_matches_declared_formula(self):
        counts = [[4, 1, 0], [0, 2, 1]]
        expected = class_tfidf(counts)
        observed = ClassTfidfTransformer().fit_transform(csr_matrix(counts)).toarray()
        self.assertEqual(expected["average_length"], 4)
        np.testing.assert_allclose(observed, expected["weights"], atol=1e-12)

    def test_invalid_label_macro_f1_matches_independent_library(self):
        gold, predicted = [0, 0, 1, 1, 1], [0, 1, 1, -1, 1]
        observed = classification_metrics(gold, predicted)["macro_f1"]
        self.assertAlmostEqual(
            observed,
            f1_score(gold, predicted, labels=[0, 1], average="macro", zero_division=0),
        )

    def test_encoder_geometry_and_saved_head_match_independent_calculations(self):
        vectors = Encoder().encode(
            [
                "A thoughtful and enjoyable movie.",
                "A dull movie with an incoherent plot.",
                "A bridge between speech recognition and translated summaries.",
            ]
        )
        numpy_scores = vectors @ vectors.T
        reference = np.einsum("ik,jk->ij", vectors, vectors, optimize=False)
        tensor_scores = (
            torch.from_numpy(vectors) @ torch.from_numpy(vectors).T
        ).numpy()
        self.assertTrue(np.isfinite(numpy_scores).all())
        np.testing.assert_allclose(numpy_scores, reference, atol=1e-6)
        np.testing.assert_allclose(numpy_scores, tensor_scores, atol=1e-6)
        saved = json.loads(
            (ROOT / "examples/artifacts/e4-classification/classifier.json").read_text()
        )
        row = np.array(saved["coef"][0])
        z = np.einsum("ij,j->i", vectors, row) + saved["intercept"][0]
        via_blas = (vectors @ row) + saved["intercept"][0]
        np.testing.assert_allclose(z, via_blas, atol=1e-10)

    def test_lightweight_reload_keeps_topics_and_uses_documented_cosine_rule(self):
        path = ROOT / "examples/artifacts/e4-topics"
        recorded = json.loads((path / "results.json").read_text())
        backend = PinnedBackend()
        model = BERTopic.load(str(path / "model"), embedding_model=backend)
        self.assertEqual(model.topics_, [row["topic"] for row in recorded["documents"]])
        selected = recorded["documents"][:3]
        vectors = backend.embed([r["text"] for r in selected])
        topic_vectors = np.asarray(model.topic_embeddings_)
        normalized = topic_vectors / np.maximum(
            np.linalg.norm(topic_vectors, axis=1, keepdims=True), 1e-12
        )
        scores = np.einsum("ik,jk->ij", vectors, normalized, optimize=False)
        expected = scores.argmax(1) - int(
            any(r["id"] == -1 for r in recorded["topics"])
        )
        actual, _ = model.transform([r["text"] for r in selected], embeddings=vectors)
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
