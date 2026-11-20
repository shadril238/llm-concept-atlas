"""Behavioral contracts for E4; all run without model downloads."""

import importlib.util
import json
from math import log
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/foundations/src"))
from atlas_foundations.e4.metrics import (
    classification_metrics,
    parse_label,
    selective_metrics,
)
from atlas_foundations.e4.topic_math import class_tfidf, mmr
from atlas_foundations.e4.hosted_results import decode_label, evaluate_external


class DiscoveryContracts(unittest.TestCase):
    def test_invalid_predictions_remain_errors_and_class_denominators(self):
        result = classification_metrics([0, 0, 1, 1, 1], [0, 1, 1, -1, 1])
        self.assertEqual(result["confusion"], [[1, 1, 0], [0, 2, 1]])
        self.assertAlmostEqual(result["accuracy"], 3 / 5)
        self.assertAlmostEqual(result["macro_f1"], 2 / 3)
        self.assertAlmostEqual(result["invalid_rate"], 1 / 5)
        self.assertEqual(classification_metrics([0, 1], [-1, -1])["macro_f1"], 0)
        for args in [([], []), ([0], [2]), ([0, 1], [1]), ([True], [0])]:
            with self.assertRaises(ValueError):
                classification_metrics(*args)

    def test_macro_and_support_weighted_averages_are_different(self):
        result = classification_metrics([0] * 9 + [1], [0] * 10)
        self.assertAlmostEqual(result["accuracy"], 0.9)
        self.assertLess(result["macro_f1"], result["weighted_f1"])
        self.assertEqual(result["per_class"][1]["recall"], 0)

    def test_output_parser_does_not_default_to_positive(self):
        self.assertEqual(
            [
                parse_label(raw)
                for raw in [
                    " positive ",
                    "NEGATIVE",
                    "not negative",
                    "",
                    None,
                    "positive because…",
                ]
            ],
            [1, 0, -1, -1, -1, -1],
        )
        self.assertEqual(decode_label('{"label":"negative"}'), 0)
        self.assertEqual(decode_label('{"label":"negative","reason":"x"}'), -1)

    def test_review_none_accepted_is_undefined_and_coverage_is_monotone(self):
        gold, predicted, scores = [0, 1, 1], [0, 0, 1], [0.9, 0.6, 0.8]
        a = selective_metrics(gold, predicted, scores, 0.6)
        b = selective_metrics(gold, predicted, scores, 0.85)
        c = selective_metrics(gold, predicted, scores, 1.0)
        self.assertEqual(a["accepted"], 3)
        self.assertEqual(b["accepted"], 1)
        self.assertIsNone(c["accepted_accuracy"])
        self.assertEqual(c["review"], 3)
        with self.assertRaises(ValueError):
            selective_metrics(gold, predicted, [float("nan")] * 3, 0.8)

    def test_ctfidf_declares_global_token_frequency_and_class_normalization(self):
        result = class_tfidf([[3, 1, 0], [0, 1, 3]])
        self.assertEqual(result["frequencies"], [3, 2, 3])
        self.assertEqual(result["average_length"], 4)
        self.assertAlmostEqual(result["weights"][0][1], log(3) / 4)
        self.assertAlmostEqual(result["weights"][0][0], 3 * log(7 / 3) / 4)
        self.assertAlmostEqual(sum(result["tf"][0]), 1)
        self.assertNotAlmostEqual(sum(result["weights"][0]), 1)
        for counts in [[], [[0, 0]], [[1, 0], [2, 0]], [[-1, 1]], [[1], [1, 2]]]:
            with self.assertRaises(ValueError):
                class_tfidf(counts)

    def test_mmr_endpoints_diversify_after_the_first_relevant_choice(self):
        terms, relevance = (
            ["translation", "translations", "evaluation"],
            [0.9, 0.88, 0.65],
        )
        similarities = [[1, 0.98, 0.2], [0.98, 1, 0.2], [0.2, 0.2, 1]]
        self.assertEqual(mmr(terms, relevance, similarities, 0, 2), terms[:2])
        self.assertEqual(
            mmr(terms, relevance, similarities, 1, 2), ["translation", "evaluation"]
        )
        self.assertEqual(len(set(mmr(terms, relevance, similarities))), 3)
        with self.assertRaises(ValueError):
            mmr(terms, relevance, similarities, 1.5)

    def test_versioned_splits_have_unique_ids_text_and_balanced_support(self):
        fixture = json.loads(
            (ROOT / "examples/data/fixtures/sentiment.json").read_text()
        )
        spec = importlib.util.spec_from_file_location(
            "prepare", ROOT / "examples/applications/discovery/prepare_sentiment.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rows = fixture["rows"]
        self.assertEqual(len({r["id"] for r in rows}), len(rows))
        self.assertEqual(len({module.canonical(r["text"]) for r in rows}), len(rows))
        for split, count in [("train", 100), ("validation", 20), ("test", 40)]:
            for label in [0, 1]:
                self.assertEqual(
                    sum(r["split"] == split and r["label"] == label for r in rows),
                    count,
                )
        self.assertEqual(sum(r["split"] == "challenge" for r in rows), 12)
        corpus = json.loads(
            (ROOT / "examples/data/fixtures/research-summaries.json").read_text()
        )
        self.assertEqual(len(corpus["documents"]), 45)
        self.assertTrue(
            all(
                set(row) == {"id", "title", "text", "source"}
                for row in corpus["documents"]
            )
        )

    def test_hosted_import_rejects_bad_ids_and_counts_missing_or_failed_outputs(self):
        fixture = {
            "rows": [
                {"id": "a", "label": 0, "split": "test"},
                {"id": "b", "label": 1, "split": "test"},
            ]
        }
        payload = {
            "provider": "test-fixture",
            "model": "not-a-live-run",
            "prompt_version": "v1",
            "collected_at": "test",
            "outputs": [{"id": "a", "raw": '{"label":"negative"}'}],
        }
        result = evaluate_external(fixture, payload)
        self.assertEqual(result["missing_outputs"], 1)
        self.assertEqual(result["metrics"]["accuracy"], 0.5)
        payload["outputs"].append({"id": "b", "raw": "positive", "error": "timeout"})
        self.assertEqual(evaluate_external(fixture, payload)["provider_errors"], 1)
        self.assertEqual(
            evaluate_external(fixture, payload)["metrics"]["invalid_rate"], 0.5
        )
        payload["outputs"].append({"id": "a", "raw": "positive"})
        with self.assertRaises(ValueError):
            evaluate_external(fixture, payload)
        payload["outputs"] = [{"id": "unknown", "raw": "positive"}]
        with self.assertRaises(ValueError):
            evaluate_external(fixture, payload)
