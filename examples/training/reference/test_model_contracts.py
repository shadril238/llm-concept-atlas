"""Run in the locked reference interpreter; requires cached BERT tokenizer only."""

import importlib.util
from collections import Counter
from pathlib import Path
import sys
import unittest
from transformers import AutoTokenizer, DataCollatorForTokenClassification

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/foundations/src"))
from atlas_foundations.ner_contract import align_first


class ActualTokenizerContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(
            "google-bert/bert-base-uncased",
            revision="86b5e0934494bd15c9632b12f734a8a67f723594",
            local_files_only=True,
            use_fast=True,
        )

    def test_split_word_specials_and_padding(self):
        encoded = self.tokenizer(
            ["Ada", "Lovelace"],
            is_split_into_words=True,
            padding="max_length",
            max_length=12,
        )
        counts = Counter(w for w in encoded.word_ids() if w is not None)
        self.assertGreater(counts[1], 1)
        labels = align_first(encoded.word_ids(), ["B-PER", "I-PER"], counts)
        encoded["labels"] = labels
        batch = DataCollatorForTokenClassification(self.tokenizer)([dict(encoded)])
        self.assertEqual(batch["labels"].shape, batch["input_ids"].shape)
        self.assertEqual((batch["labels"] != -100).sum().item(), 2)
        self.assertTrue(
            all(
                label == -100
                for word, label in zip(encoded.word_ids(), labels)
                if word is None
            )
        )

    def test_actual_truncation_and_no_entity_sentence(self):
        words = ["Ada", "unbelievableness", "visited", "New", "York"]
        complete = self.tokenizer(words, is_split_into_words=True)
        counts = Counter(w for w in complete.word_ids() if w is not None)
        partial = self.tokenizer(
            words, is_split_into_words=True, truncation=True, max_length=5
        )
        with self.assertRaises(ValueError):
            align_first(
                partial.word_ids(), ["B-PER", "O", "O", "B-LOC", "I-LOC"], counts
            )
        empty = self.tokenizer(["Rain", "fell"], is_split_into_words=True)
        self.assertEqual(
            [
                i
                for i in align_first(
                    empty.word_ids(), ["O", "O"], Counter({0: 1, 1: 1})
                )
                if i != -100
            ],
            [0, 0],
        )


if __name__ == "__main__":
    unittest.main()
