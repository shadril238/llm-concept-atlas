import test from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import {
  classificationMetrics,
  parseLabel,
  classTfidf,
  mmr,
} from "../src/lib/discovery-mechanics.js";

test("browser and Python agree on invalid-label metrics, c-TF-IDF, and MMR tradeoffs", () => {
  const run = spawnSync(
    "python3",
    [
      "-c",
      `import sys,json
sys.path.insert(0,'examples/foundations/src')
from atlas_foundations.e4.metrics import classification_metrics
from atlas_foundations.e4.topic_math import class_tfidf,mmr
print(json.dumps({'metrics':classification_metrics([0,0,1,1,1],[0,1,1,-1,1]),'ctfidf':class_tfidf([[3,1,0],[0,1,3]]),'mmr':mmr(['a','b','c'],[.9,.88,.65],[[1,.98,.2],[.98,1,.2],[.2,.2,1]],.3,2)}))`,
    ],
    { encoding: "utf8" },
  );
  assert.equal(run.status, 0, run.stderr);
  const python = JSON.parse(run.stdout);
  const browser = classificationMetrics([0, 0, 1, 1, 1], [0, 1, 1, -1, 1]);
  for (const key of [
    "confusion",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "invalid_rate",
    "per_class",
  ])
    assert.deepEqual(browser[key], python.metrics[key]);
  const weights = classTfidf([
    [3, 1, 0],
    [0, 1, 3],
  ]);
  assert.deepEqual(weights.frequencies, python.ctfidf.frequencies);
  assert.deepEqual(weights.tf, python.ctfidf.tf);
  assert.equal(weights.average_length, python.ctfidf.average_length);
  weights.idf.forEach((v, i) =>
    assert.ok(Math.abs(v - python.ctfidf.idf[i]) < 1e-12),
  );
  weights.weights.forEach((row, i) =>
    row.forEach((v, j) =>
      assert.ok(Math.abs(v - python.ctfidf.weights[i][j]) < 1e-12),
    ),
  );
  assert.deepEqual(
    mmr(
      ["a", "b", "c"],
      [0.9, 0.88, 0.65],
      [
        [1, 0.98, 0.2],
        [0.98, 1, 0.2],
        [0.2, 0.2, 1],
      ],
      0.3,
      2,
    ),
    python.mmr,
  );
});
test("invalid output and zero-count tables cannot silently become valid decisions", () => {
  assert.deepEqual(
    ["POSITIVE", " negative ", "not negative", "positive because…", ""].map(
      parseLabel,
    ),
    [1, 0, -1, -1, -1],
  );
  assert.throws(() =>
    classTfidf([
      [0, 0],
      [1, 2],
    ]),
  );
  assert.throws(() => classificationMetrics([], []));
  assert.throws(() => mmr(["a"], [1], [[1]], 2, 1));
});

test("generated labels retain the verified parent topic artifact", () => {
  const published = JSON.parse(
    readFileSync("src/data/discovery-results.json", "utf8"),
  );
  const record = JSON.parse(
    readFileSync("examples/verification/records/modular-bertopic.json", "utf8"),
  );
  assert.equal(record.status, "passed");
  assert.equal(
    published.generation.parent_artifact_sha256,
    record.artifactSha256,
  );
  assert.deepEqual(
    published.generation.labels.map((row) => row.topic).sort((a, b) => a - b),
    published.topics.topics.map((row) => row.id).sort((a, b) => a - b),
  );
});
