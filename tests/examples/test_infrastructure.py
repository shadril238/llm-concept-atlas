import copy
import importlib.util
import json
import re
from pathlib import Path
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from example_tools import ROOT, load, region, validate


class InfrastructureTests(unittest.TestCase):
    def setUp(self):
        self.manifest = load()

    def test_manifest_and_all_published_blocks_are_registered(self):
        self.assertEqual(validate(self.manifest), [])
        published = []
        for path in (ROOT / "src/content/docs").rglob("*.mdx"):
            published.extend(re.findall(r"\*\*Expected output:\*\*\n\n```text\n([\s\S]*?)\n```", path.read_text()))
        registered = [entry["validation"]["expected"] for entry in self.manifest["examples"]
                      if entry["validation"].get("origin") == "existing-published-output"]
        self.assertCountEqual(registered, [value.strip() for value in published])

    def test_missing_registration_and_source_drift_fail(self):
        self.manifest["examples"].pop(0)
        self.assertTrue(any("unregistered" in e for e in validate(self.manifest)))
        self.manifest = load()
        self.manifest["examples"][0]["lesson"]["block"] = 1000
        self.assertTrue(any("drift" in e for e in validate(self.manifest)))

    def test_duplicate_ids_and_unknown_sources_fail(self):
        self.manifest["examples"].append(copy.deepcopy(self.manifest["examples"][0]))
        self.manifest["examples"][0]["sourceSections"] = ["missing"]
        errors = validate(self.manifest)
        self.assertTrue(any("duplicate ID" in e for e in errors))
        self.assertTrue(any("unknown source" in e for e in errors))

    def test_models_require_pinned_revisions(self):
        entry = next(
            e for e in self.manifest["examples"] if e["id"] == "embedding-retrieval"
        )
        entry["models"][0]["revision"] = "main"
        self.assertTrue(any("revision" in e and "pattern" in e for e in validate(self.manifest)))

    def test_bad_entrypoint_cannot_escape_repository(self):
        self.manifest["examples"][0]["entrypoint"] = "../outside.py"
        self.assertTrue(any("unsafe entrypoint" in e for e in validate(self.manifest)))

    def test_verification_cannot_survive_contract_change(self):
        entry = next(
            e for e in self.manifest["examples"] if e["id"] == "attention-softmax"
        )
        entry["validation"]["expected"] = "incorrect result"
        self.assertTrue(
            any("stale verification contract" in e for e in validate(self.manifest))
        )

    def test_attention_normalization_translation_and_invalid_inputs(self):
        path = ROOT / "examples/foundations/src/atlas_foundations/attention.py"
        spec = importlib.util.spec_from_file_location("attention", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        weights, result = module.attend([2.0, 1.0], [10.0, 0.0])
        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertAlmostEqual(result, 7.3105857863)
        self.assertEqual(weights, module.attend([1002.0, 1001.0], [10.0, 0.0])[0])
        for scores, values in [([], []), ([1.0], []), ([float("nan")], [1.0])]:
            with self.assertRaises(ValueError):
                module.attend(scores, values)

    def test_generated_foundation_notebook_runs_in_fresh_process(self):
        path = ROOT / "examples/foundations/notebooks/attention-softmax.ipynb"
        notebook = json.loads(path.read_text())
        code = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        run = subprocess.run(
            [str(ROOT / "examples/foundations/.venv/bin/python"), "-c", code],
            cwd=path.parent,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("7.3106", run.stdout)

    def test_model_execution_requires_explicit_network_choice(self):
        run = subprocess.run(
            [sys.executable, "scripts/run-examples.py", "--id", "embedding-retrieval"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("choose --allow-network", run.stderr)


if __name__ == "__main__":
    unittest.main()
