"""Behavioral checks for masks, shapes, spans, and reference examples."""

from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/foundations/src"))
from atlas_foundations.ner_contract import LABELS, align_first, entity_scores, spans


class ReferenceContracts(unittest.TestCase):
    def test_first_piece_masks_special_padding_and_continuations(self):
        ids = [None, 0, 1, 1, None, None]
        self.assertEqual(
            align_first(ids, ["B-PER", "I-PER"], Counter({0: 1, 1: 2})),
            [-100, 1, 2, -100, -100, -100],
        )

    def test_partial_word_missing_word_and_partial_entity_are_rejected(self):
        for ids in [[None, 0, 1, None], [None, 0, None]]:
            with self.assertRaisesRegex(ValueError, "Truncated"):
                align_first(ids, ["B-PER", "I-PER"], Counter({0: 1, 1: 2}))

    def test_spans_keep_multiword_entities_and_repair_invalid_i(self):
        self.assertEqual(
            spans(["B-PER", "I-PER", "O", "B-ORG"]), [("PER", 0, 2), ("ORG", 3, 4)]
        )
        self.assertEqual(
            spans(["I-PER", "I-PER", "I-ORG"]), [("PER", 0, 2), ("ORG", 2, 3)]
        )
        self.assertEqual(spans(["O", "O"]), [])

    def test_sentence_boundaries_and_manual_type_boundary_errors(self):
        gold = [["B-PER", "I-PER"], ["B-ORG", "I-ORG"]]
        correct = entity_scores(gold, gold)
        self.assertEqual(correct["f1"], 1)
        wrong = entity_scores(gold, [["B-PER", "O"], ["B-LOC", "I-LOC"]])
        self.assertEqual(wrong["tp"], 0)
        self.assertEqual(
            entity_scores([["B-PER"], ["I-PER"]], [["B-PER"], ["I-PER"]])["gold"], 2
        )
        self.assertEqual(entity_scores([["O"]], [["O"]])["f1"], 0)
        with self.assertRaises(ValueError):
            entity_scores([["O"]], [[]])

    def test_split_names_are_disjoint_and_fixtures_never_duplicate_text(self):
        data = json.loads(
            (ROOT / "examples/data/fixtures/ner-reference.json").read_text()
        )["records"]
        groups = {split: set() for split in ["train", "validation", "test"]}
        texts = set()
        for row in data:
            text = tuple(row["tokens"])
            self.assertNotIn(text, texts)
            texts.add(text)
            self.assertEqual(len(row["tags"]), len(row["tokens"]))
            for _, start, end in spans(row["tags"]):
                groups[row["split"]].add(tuple(row["tokens"][start:end]))
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["train"] & groups["validation"])
        self.assertFalse(groups["test"] & groups["validation"])
