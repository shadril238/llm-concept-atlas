"""Behavioral regressions for silent data/mask/reference/judgment mistakes."""

import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/foundations/src"))
from atlas_foundations.e8.contracts import *


class Tokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


class GenerationContracts(unittest.TestCase):
    def test_fresh_notebook_record_bindings(self):
        import hashlib
        project=ROOT/'examples/training/generation'
        record=json.loads((project/'notebook-execution.json').read_text())
        self.assertEqual(len(record['records']),3)
        self.assertEqual(record['verifierSha256'],hashlib.sha256((project/'verify_notebooks.py').read_bytes()).hexdigest())
        self.assertEqual(record['lockSha256'],hashlib.sha256((project/'uv.lock').read_bytes()).hexdigest())
        for path,digest in record['sourceHashes'].items():
            self.assertEqual(digest,hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),path)
        for row in record['records']:
            self.assertEqual(row['status'],'passed')
            self.assertEqual(row['codeCells'],4)
            self.assertEqual(row['sha256'],hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest())

    def row(self):
        return dict(
            id="one",
            split="train",
            instruction="Extract.",
            context="Code: A",
            response="A",
            chosen="A",
            rejected="B",
        )

    def test_prompt_padding_eos_mask(self):
        r = self.row()
        b = encode(Tokenizer(), r, max_length=500)
        self.assertEqual(b["labels"][: b["prefix_tokens"]], [-100] * b["prefix_tokens"])
        self.assertEqual(b["labels"][-2:], [ord("A"), 0])
        self.assertTrue(all(b["attention_mask"]))

    def test_overflow_and_empty_fail(self):
        with self.assertRaises(ValueError):
            encode(Tokenizer(), self.row(), max_length=3)
        with self.assertRaises(AssertionError):
            encode(Tokenizer(), self.row(), response="")

    def test_leakage_and_invalid_pairs_fail(self):
        r = self.row()
        with self.assertRaises(AssertionError):
            check_data([r, dict(r, id="two", split="test")])
        with self.assertRaises(AssertionError):
            check_preferences([dict(r, rejected="A")])

    def test_dpo_reference_and_shift_invariance(self):
        self.assertAlmostEqual(dpo_loss(-2, -4, -2, -4), math.log(2))
        self.assertLess(dpo_loss(-1, -4, -2, -4), math.log(2))
        self.assertAlmostEqual(dpo_loss(-1, -4, -2, -4), dpo_loss(-11, -14, -22, -24))
        with self.assertRaises(ValueError):
            dpo_loss(1, 2, 3, 4, beta=0)

    def test_importer_rejects_stale_duplicate_and_reports_missing(self):
        source = [
            dict(model="base", id="a", prediction="A"),
            dict(model="dpo", id="a", prediction="B"),
        ]
        rating = dict(
            model="base",
            id="a",
            output_sha256=output_hash("A"),
            rubric_version="atlas-e8-v1",
            judge_version="author-test-v1",
            judge_prompt_version="rubric-v1",
            judge_prompt_sha256="a" * 64,
            order="AB",
            grounding=2,
            instruction=2,
            clarity=2,
        )
        self.assertFalse(validate_judgments(source, [rating])["complete"])
        with self.assertRaises(ValueError):
            validate_judgments(source, [dict(rating, output_sha256="stale")])
        with self.assertRaises(ValueError):
            validate_judgments(source, [rating, rating])
        with self.assertRaises(ValueError):
            validate_judgments(source, [dict(rating, grounding=True)])

    def test_actual_fixture_splits(self):
        rows = json.loads(
            (ROOT / "examples/data/fixtures/instruction-dolly.json").read_text()
        )["rows"]
        self.assertTrue(check_data(rows))
        self.assertEqual(
            [
                sum(r["split"] == s for r in rows)
                for s in ["train", "validation", "test"]
            ],
            [32, 8, 8],
        )
        self.assertTrue(
            check_preferences(
                json.loads(
                    (
                        ROOT / "examples/data/fixtures/generation-preferences.json"
                    ).read_text()
                )["rows"]
            )
        )
