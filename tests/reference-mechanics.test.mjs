import test from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import {
  bioSpans,
  loraTrace,
  spanScore,
} from "../src/lib/reference-mechanics.js";

test("LoRA explorer reproduces worked trace and alpha intervention", () => {
  const result = loraTrace(1, 1);
  assert.equal(result.compressed, 0.5);
  assert.deepEqual(result.output, [2.05, 1.1]);
  assert.deepEqual(loraTrace(0, 1).output, [2, 1]);
  assert.deepEqual(loraTrace(2, 2).output, result.output);
});
test("browser span decoder agrees with Python on sentence-level BIO cases", () => {
  const cases = [
    ["B-PER", "I-PER", "O", "B-ORG"],
    ["I-PER", "I-PER", "I-ORG"],
    ["O", "O"],
    ["B-ORG", "B-ORG"],
  ];
  const code = `import json,sys;sys.path.insert(0,'examples/foundations/src');from atlas_foundations.ner_contract import spans;print(json.dumps([spans(tags) for tags in json.loads(sys.argv[1])]))`;
  const run = spawnSync("python3", ["-c", code, JSON.stringify(cases)], {
    encoding: "utf8",
  });
  assert.equal(run.status, 0, run.stderr);
  assert.deepEqual(cases.map(bioSpans), JSON.parse(run.stdout));
  assert.equal(
    spanScore(["B-PER", "O", "B-ORG", "I-ORG"], ["B-PER", "O", "B-ORG", "O"])
      .f1,
    0.5,
  );
});
