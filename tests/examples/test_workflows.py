"""E5 behavioral boundaries, offline and independent of model/framework installs."""

import json
import hashlib
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/foundations/src"))
from atlas_foundations.e5.contracts import (
    validate_order,
    extract_answer,
    vote,
    calculator,
    dispatch,
    bounded_loop,
)
from atlas_foundations.e5.search import search, verify as verify_expression
from atlas_foundations.e5.memory import pack


class WorkflowContracts(unittest.TestCase):
    def test_strict_order_shape_is_separate_from_content(self):
        good = '{"item":"pen","quantity":3,"priority":"urgent"}'
        self.assertTrue(validate_order(good)["valid"])
        for bad in [
            good + " prose",
            good.replace("3", "true"),
            good.replace("3", "3.0"),
            good.replace("3", "1e0"),
            good.replace("3", "10"),
            good.replace("3", "0"),
            good.replace("pen", "pencil"),
            good.replace("3", '3,"quantity":8'),
            good.replace('"item"', '"it\\u0065m"').replace("3", '3,"item":"book"'),
        ]:
            self.assertFalse(validate_order(bad)["valid"], bad)
        wrong = validate_order(good.replace("3", "8"))
        self.assertTrue(wrong["valid"])
        self.assertNotEqual(wrong["value"]["quantity"], 3)

    def test_answer_parser_requires_complete_answer_contract(self):
        self.assertEqual(extract_answer(" 15 "), 15)
        self.assertEqual(extract_answer("7+8=15\nAnswer: 15", "answer-line"), 15)
        for raw in [
            "The answer is 15",
            "15 books",
            "Answer: 15 and more",
            "prefix Answer: 15",
            "Answer: 15\nexplanation",
        ]:
            self.assertIsNone(extract_answer(raw, "answer-line"))

    def test_votes_do_not_discard_invalids_or_break_ties(self):
        result = vote([15, 15, 14, None, None])
        self.assertEqual(result["winner"], 15)
        self.assertFalse(result["strict_majority"])
        self.assertEqual(result["invalid"], 2)
        self.assertIsNone(vote([15, 14, None])["winner"])
        self.assertIsNone(vote([None, None])["winner"])
        self.assertTrue(vote([15, 15, None])["strict_majority"])
        self.assertEqual(vote([True, 15])["invalid"], 1)
        with self.assertRaises(ValueError):
            vote([])

    def test_exact_fractional_solution_and_leaf_accounting(self):
        self.assertTrue(verify_expression("8/(3-8/3)", [3, 3, 8, 8]))
        for expression in ["24", "8/(3-8/3)+0", '__import__("os")', "8**3", "1/0"]:
            self.assertFalse(verify_expression(expression, [3, 3, 8, 8]))
        complete = search([3, 3, 8, 8])
        self.assertTrue(complete["verified"])
        self.assertEqual(complete["stop_reason"], "solution")
        limited = search([3, 3, 8, 8], max_nodes=16)
        self.assertEqual(limited["stop_reason"], "node_budget")
        self.assertFalse(limited["verified"])
        impossible = search([1, 1, 1, 1])
        self.assertEqual(impossible["stop_reason"], "frontier_exhausted")
        narrow = search([2, 3, 4, 6], beam=1)
        self.assertFalse(narrow["verified"])
        self.assertGreater(narrow["beam_pruned"], 0)

    def test_tool_arguments_are_bounded_and_actions_allowlisted(self):
        self.assertEqual(calculator("divide", 9, 3), 3)
        for op, left, right in [
            ("add", True, 3),
            ("divide", 3, 0),
            ("add", float("inf"), 1),
            ("add", 1000001, 0),
            ("exec", 2, 3),
        ]:
            with self.assertRaises(ValueError):
                calculator(op, left, right)
        for action in [
            {"tool": "shell", "arguments": {}},
            {"tool": "lookup", "arguments": {"key": "private"}},
            {"tool": "lookup", "arguments": {"key": "hours", "extra": 1}},
        ]:
            with self.assertRaises(ValueError):
                dispatch(action, {"hours": "17:30"})

    def test_controller_observes_errors_and_stops_at_budget(self):
        scripted = iter(
            [
                '{"tool":"calculator","arguments":{"operation":"add","left":true,"right":8}}',
                '{"tool":"calculator","arguments":{"operation":"add","left":7,"right":8}}',
                '{"final":"15"}',
            ]
        )

        def choose(question, trace):
            return next(scripted)

        out = bounded_loop(choose, lambda a: dispatch(a, {}), "add", 3)
        self.assertEqual(
            [t["kind"] for t in out["trace"]], ["invalid", "tool", "final"]
        )
        self.assertEqual(out["trace"][1]["observation"], "15")
        out = bounded_loop(
            lambda q, t: '{"tool":"lookup","arguments":{"key":"hours"}}',
            lambda a: dispatch(a, {"hours": "17:30"}),
            "repeat",
            3,
        )
        self.assertEqual(out["stop_reason"], "max_steps")
        self.assertEqual(len(out["trace"]), 3)
        self.assertIsNone(out["answer"])
        for n in [0, 11, True]:
            with self.assertRaises(ValueError):
                bounded_loop(choose, lambda a: 0, "", n)

    def test_memory_removes_whole_pairs_and_accounts_for_reserve(self):
        class Counter:
            def count(self, messages):
                return sum(len(m["content"].split()) for m in messages)

        history = [
            {"id": f"{r}{i}", "role": role, "content": ("word " * 12).strip()}
            for i in range(4)
            for r, role in [("u", "user"), ("a", "assistant")]
        ]
        model = Counter()
        r = pack(model, history, "question", "buffer", "", 110, 20)
        self.assertLessEqual(r["input_tokens"] + 20, 110)
        self.assertEqual(len(r["dropped_for_budget"]) % 2, 0)
        w = pack(model, history, "question", "window", "", 500, 20)
        self.assertEqual(w["retained_ids"], [h["id"] for h in history[-4:]])
        self.assertEqual(len(w["omitted_by_policy"]), 4)
        s = pack(model, history, "question", "summary", "long " * 300, 110, 20)
        self.assertFalse(s["summary_kept"])
        with self.assertRaises(ValueError):
            pack(model, history, "question", "buffer", "", 1, 20)

    def test_memory_rejects_unknown_policy_and_incomplete_turns(self):
        class Counter:
            def count(self, messages):
                return 1

        for history, policy, reserve in [
            ([], "typo", 64),
            ([{"role": "user", "id": "u1", "content": "x"}], "buffer", 64),
            ([], "buffer", True),
            ([], "buffer", -1),
        ]:
            with self.assertRaises(ValueError):
                pack(Counter(), history, "question", policy, "", 192, reserve)

    def test_notebook_run_all_evidence_matches_sources_and_lock(self):
        project = ROOT / "examples/applications/workflows"
        evidence = json.loads((project / "notebook-execution.json").read_text())
        self.assertEqual(
            evidence["lockSha256"],
            hashlib.sha256((project / "uv.lock").read_bytes()).hexdigest(),
        )
        self.assertEqual(
            evidence["verifierSha256"],
            hashlib.sha256((project / "verify_notebooks.py").read_bytes()).hexdigest(),
        )
        self.assertEqual(len(evidence["records"]), 5)
        for record in evidence["records"]:
            path = ROOT / record["path"]
            self.assertEqual(
                record["sha256"], hashlib.sha256(path.read_bytes()).hexdigest()
            )
            self.assertEqual(record["status"], "passed")
            self.assertEqual(record["codeCells"], 4)

    def test_fixture_splits_are_disjoint_and_references_not_gold_domains(self):
        fixture = json.loads(
            (ROOT / "examples/data/fixtures/workflow-tasks.json").read_text()
        )
        for field in ["arithmetic", "extraction"]:
            rows = fixture[field]
            self.assertEqual(len({r["id"] for r in rows}), len(rows))
            design = {r["id"] for r in rows if r["split"] == "design"}
            evaluation = {r["id"] for r in rows if r["split"] == "evaluation"}
            self.assertTrue(design.isdisjoint(evaluation))


if __name__ == "__main__":
    unittest.main()
