import test from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import {
  crossEntropy,
  multipleNegatives,
  pairFeatures,
  cosineRegression,
  supervisedMask,
} from "../src/lib/training-mechanics.js";
const close = (a, b) => assert.ok(Math.abs(a - b) < 1e-9, `${a} != ${b}`);
test("stable cross entropy agrees across Python and browser including large offsets", () => {
  const cases = [
    [[4, 2], 0],
    [[1004, 1002], 0],
    [[-1000, -1001, -1002], 2],
  ];
  const run = spawnSync(
    "python3",
    [
      "-c",
      "import sys,json;sys.path.insert(0,'examples/foundations/src');from atlas_foundations.e7.contracts import cross_entropy;print(json.dumps([cross_entropy(*r)for r in json.load(sys.stdin)]))",
    ],
    { input: JSON.stringify(cases), encoding: "utf8" },
  );
  assert.equal(run.status, 0, run.stderr);
  JSON.parse(run.stdout).forEach((v, i) => close(v, crossEntropy(...cases[i])));
  assert.throws(() => crossEntropy([], 0));
  assert.throws(() => crossEntropy([1, 2], true));
  assert.throws(() => crossEntropy([NaN, 2], 0));
});
test("directional MNR uses its declared row targets and matches Python independently", () => {
  const scores = [
      [0.8, 0.4, 0.2],
      [0.3, 0.9, 0.1],
      [0.5, 0.2, 0.7],
    ],
    scale = 5;
  const run = spawnSync(
    "python3",
    [
      "-c",
      "import sys,json;sys.path.insert(0,'examples/foundations/src');from atlas_foundations.e7.contracts import mnr_loss;print(json.dumps(mnr_loss(**json.load(sys.stdin))))",
    ],
    { input: JSON.stringify({ scores, scale }), encoding: "utf8" },
  );
  assert.equal(run.status, 0, run.stderr);
  close(multipleNegatives(scores, scale).loss, JSON.parse(run.stdout).loss);
  assert.throws(() =>
    multipleNegatives([
      [1, 2, 3],
      [2, 3, 4],
    ]),
  );
  assert.throws(() =>
    multipleNegatives(
      [
        [1, 2],
        [2, 1],
      ],
      0,
    ),
  );
});
test("pair features and regression keep their different label contracts", () => {
  const f = pairFeatures([1, 0.5], [0.2, 0.8]);
  assert.equal(f.length, 6);
  close(f[4], 0.8);
  close(f[5], 0.3);
  close(cosineRegression([0.8], [1]).loss, 0.04);
  close(cosineRegression([0.8], [0]).loss, 0.64);
  assert.throws(() => pairFeatures([1], [1, 2]));
  assert.throws(() => cosineRegression([], []));
});
test("MLM selected unchanged tokens retain targets and specials/padding cannot be selected", () => {
  const tokens = [
    { original_id: 101, special_or_padding: true },
    { original_id: 2000, special_or_padding: false },
    { original_id: 3000, special_or_padding: false },
    { original_id: 0, special_or_padding: true },
  ];
  assert.deepEqual(supervisedMask(tokens, [1]), [-100, 2000, -100, -100]);
  assert.deepEqual(supervisedMask(tokens, []), [-100, -100, -100, -100]);
  assert.throws(() => supervisedMask(tokens, [0]));
  assert.throws(() => supervisedMask(tokens, [1, 1]));
  assert.throws(() => supervisedMask(tokens, [true]));
});
test("published independent arithmetic checks match actual model-library losses", () => {
  const d = JSON.parse(readFileSync("src/data/training-results.json", "utf8"))
    .embedding.loss_checks;
  close(crossEntropy(d.softmax.logits, d.softmax.target), d.softmax.manual);
  assert.ok(
    Math.abs(
      cosineRegression(d.cosine.scores, d.cosine.targets).loss -
        d.cosine.manual,
    ) < 1e-6,
    "float64 browser vs float32 tensor reduction",
  );
  close(multipleNegatives(d.mnr.scores, d.mnr.scale).loss, d.mnr.manual);
  for (const r of Object.values(d))
    assert.ok(Math.abs(r.manual - r.library) < 2e-5);
});
test("all actual ranks and all-seed statistics recompute from underlying predictions", () => {
  const d = JSON.parse(readFileSync("src/data/training-results.json", "utf8"));
  for (const r of d.embedding.conditions) {
    const mrr =
      r.retrieval.rows.reduce((s, p) => s + 1 / p.rank, 0) /
      r.retrieval.rows.length;
    close(mrr, r.retrieval.mrr);
    assert.equal(r.training.reload_verified, true);
  }
  for (const summary of d.classification.summary) {
    const values = d.classification.runs
      .filter((r) => r.method === summary.method)
      .map((r) => r.test.macro_f1);
    const mean = values.reduce((a, b) => a + b, 0) / 3;
    close(mean, summary.mean_macro_f1);
    close(
      Math.sqrt(values.reduce((s, x) => s + (x - mean) ** 2, 0) / 3),
      summary.std_macro_f1,
    );
  }
});
