import test from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import {
  aggregateAnswers,
  validateOrder,
} from "../src/lib/workflow-mechanics.js";
const objects = [
  '{"item":"pen","quantity":3,"priority":"urgent"}',
  '{"item":"book","quantity":9,"priority":"normal"}',
  '{"item":"pen","quantity":3.0,"priority":"urgent"}',
  '{"item":"pen","quantity":1e0,"priority":"urgent"}',
  '{"item":"pen","quantity":true,"priority":"urgent"}',
  '{"item":"pen","quantity":3,"quantity":8,"priority":"urgent"}',
  '{"item":"pen","quantity":3,"quanti\\u0074y":8,"priority":"urgent"}',
  '{"item":"pen","quantity":0,"priority":"urgent"}',
  '{"item":"pen","quantity":3,"priority":"urgent","extra":1}',
  '{"item":"pen","quantity":3,"priority":"urgent"} prose',
  '{"item":"pen","quantity":{"nested":3},"priority":"urgent"}',
  "[]",
  "null",
  '{"item":"book","quantity":9,"priority":"a\\\"b"}',
];
test("strict browser and Python validators agree on ambiguity, type, range, and extra prose", () => {
  const run = spawnSync(
    "python3",
    [
      "-c",
      "import sys,json;sys.path.insert(0,'examples/foundations/src');from atlas_foundations.e5.contracts import validate_order;print(json.dumps([validate_order(x)['valid'] for x in json.load(sys.stdin)]))",
    ],
    { input: JSON.stringify(objects), encoding: "utf8" },
  );
  assert.equal(run.status, 0, run.stderr);
  assert.deepEqual(
    objects.map((x) => validateOrder(x).valid),
    JSON.parse(run.stdout),
  );
});
test("plurality and strict majority agree across browser and Python, including invalids and ties", () => {
  const cases = [
    [15, 15, 14, null, null],
    [15, 14, null],
    [null, null],
    [15, 15, null],
    [true, 15],
    [14, 14, 14, 15, 15],
  ];
  const run = spawnSync(
    "python3",
    [
      "-c",
      "import sys,json;sys.path.insert(0,'examples/foundations/src');from atlas_foundations.e5.contracts import vote;print(json.dumps([vote(x) for x in json.load(sys.stdin)]))",
    ],
    { input: JSON.stringify(cases), encoding: "utf8" },
  );
  assert.equal(run.status, 0, run.stderr);
  const py = JSON.parse(run.stdout);
  cases.forEach((x, i) => {
    const js = aggregateAnswers(x);
    for (const key of ["winner", "invalid", "strict_majority"])
      assert.equal(js[key], py[i][key]);
  });
});
test("published native generation, memory budgets, search ancestry, and bounded traces retain evidence", () => {
  const d = JSON.parse(readFileSync("src/data/workflow-results.json", "utf8"));
  assert.equal(
    d.prompts.extraction.filter(
      (r) => r.variant === "native-schema" && r.validation.valid,
    ).length,
    9,
  );
  assert.ok(
    d.prompts.extraction.some((r) => r.validation.valid && !r.answer_correct),
  );
  assert.equal(d.prompts.truncation_probe.finish_reason, "length");
  assert.equal(validateOrder(d.prompts.truncation_probe.raw).valid, false);
  for (const r of d.memory.cases) {
    assert.ok(
      r.packed.input_tokens + r.packed.reserved_output_tokens <= r.budget,
    );
    assert.equal(r.call.prompt_tokens, r.packed.input_tokens);
    assert.equal(r.packed.dropped_for_budget.length % 2, 0);
  }
  for (const r of [...d.search.games, d.search.guided])
    for (const id of r.solution_path)
      assert.ok(r.display_nodes.some((n) => n.id === id));
  for (const r of d.agents.real_cases) {
    assert.ok(r.outcome.trace.length <= 3);
    assert.equal(r.outcome.trace[0].kind, "tool");
    assert.ok(r.decisions.length > 0);
  }
  assert.equal(d.chains.equivalent_prompt_checked, true);
});
