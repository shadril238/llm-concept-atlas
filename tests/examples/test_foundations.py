"""E3 behavioral invariants, not a mirror of one output snapshot."""

import importlib.util
import json
from math import isclose
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/foundations/src"))
from atlas_foundations.e3.mechanics import (
    attention,
    cosine,
    filter_distribution,
    kv_bytes,
    normalize,
    pairs,
    rope_pair,
    sample,
    softmax,
)


class FoundationMechanics(unittest.TestCase):
    def test_softmax_shift_and_sampling_contract(self):
        self.assertEqual(softmax([2.0, 1.0, 0.0]), softmax([1002.0, 1001.0, 1000.0]))
        for args in [([], 1), ([1], 0), ([float("inf")], 1)]:
            with self.assertRaises(ValueError):
                softmax(*args)
        filtered = filter_distribution([2.0, 1.0, 0.0], top_p=0.8)
        self.assertEqual(filtered[2], 0)
        self.assertAlmostEqual(sum(filtered), 1)
        self.assertEqual(filter_distribution([2.0, 1.0, 0.0], top_k=1), [1.0, 0.0, 0.0])
        self.assertEqual(sample(filtered, 11), sample(filtered, 11))
        self.assertLess(
            max(softmax([2.0, 1.0, 0.0], 2)), max(softmax([2.0, 1.0, 0.0], 0.5))
        )

    def test_mask_future_exclusion_and_packed_boundary(self):
        q = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        values = [[1.0, 0.0], [0.0, 2.0], [3.0, 1.0]]
        weights, output = attention(q, q, values)
        changed = [values[0], values[1], [999.0, 999.0]]
        self.assertEqual(output[:2], attention(q, q, changed)[1][:2])
        self.assertEqual(weights[0], [1.0, 0.0, 0.0])
        packed, _ = attention(q, q, values, segments=[0, 0, 1])
        self.assertEqual(packed[2], [0.0, 0.0, 1.0])

    def test_rotation_norm_and_relative_offset(self):
        q, k = [1.0, 2.0], [3.0, -0.5]
        dot = lambda a, b: sum(x * y for x, y in zip(a, b))
        rotated = rope_pair(q, 4, 0.5)
        self.assertAlmostEqual(dot(q, q), dot(rotated, rotated))
        self.assertAlmostEqual(
            dot(rope_pair(q, 2, 0.5), rope_pair(k, 5, 0.5)),
            dot(rope_pair(q, 9, 0.5), rope_pair(k, 12, 0.5)),
        )
        self.assertNotEqual(
            normalize([1.0, 2.0, 3.0]), normalize([1.0, 2.0, 3.0], rms=True)
        )

    def test_pairs_and_cache_count(self):
        self.assertEqual(
            pairs(["a", "b", "c"]), [("a", "b"), ("b", "a"), ("b", "c"), ("c", "b")]
        )
        self.assertEqual(kv_bytes(6, 1, 4, 12, 32, 8), 147456)
        with self.assertRaises(ValueError):
            cosine([0.0, 0.0], [1.0, 0.0])

    def test_playlist_sets_and_splits_are_disjoint(self):
        data = json.loads((ROOT / "examples/data/fixtures/playlists.json").read_text())
        seen = set()
        for row in data["playlists"]:
            songs = frozenset(row["songs"])
            self.assertNotIn(songs, seen)
            seen.add(songs)
            self.assertEqual(len(row["songs"]), len(songs))
        train = {row["id"] for row in data["playlists"] if row["split"] == "train"}
        test = {row["id"] for row in data["playlists"] if row["split"] == "test"}
        self.assertFalse(train & test)
        self.assertTrue(all("cold" not in row["songs"] for row in data["playlists"]))

    def test_optional_event_importer_keeps_user_groups_and_sessions(self):
        path = ROOT / "examples/training/reference/import_lastfm.py"
        spec = importlib.util.spec_from_file_location("import_lastfm", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        lines = []
        for day in [1, 2]:
            for minute in range(4):
                lines.append(
                    f"user-a\t2020-01-{day:02}T00:{minute:02}:00Z\tartist\tArtist\tt{minute + day * 10}\tSong {minute}\n"
                )
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / "events.tsv"
            file.write_text("".join(reversed(lines)))
            result = module.convert(file)
        self.assertEqual(len(result["playlists"]), 2)
        self.assertEqual(len({row["split"] for row in result["playlists"]}), 1)
        self.assertNotIn("user-a", json.dumps(result))
