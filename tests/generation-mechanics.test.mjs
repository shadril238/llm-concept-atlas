import test from "node:test";
import assert from "node:assert/strict";
import {
  lowRank,
  trainingMemory,
  preferenceLoss,
  lossMask,
} from "../src/lib/generation-mechanics.js";
test("LoRA dimensions include both projections and alpha/r scaling", () => {
  assert.deepEqual(lowRank(128, 384, 4, 8), {
    base: 49152,
    adapter: 2048,
    scale: 2,
  });
  assert.equal(lowRank(128, 384, 8, 8).adapter, 4096);
});
test("weights, trainable storage, gradients and moments are separate terms", () => {
  const rows = trainingMemory(1000, 10, 4, 3);
  assert.equal(rows[0].mib * 2 ** 20, 500);
  assert.equal(rows[3].mib * 2 ** 20, 80);
  assert.throws(() => trainingMemory(100, 101));
});
test("DPO relative reference shift and exact initialization tie", () => {
  assert.equal(preferenceLoss(-2, -4, -2, -4, 0.1).margin, 0);
  assert.equal(preferenceLoss(-2, -4, -2, -4, 0.1).loss, Math.log(2));
  assert.ok(preferenceLoss(-1, -4, -2, -4, 0.1).loss < Math.log(2));
  assert.ok(Number.isFinite(preferenceLoss(-10000, 0, 0, 0, 1).loss));
});
test("real EOS zero remains supervised while padding zero is excluded", () => {
  assert.deepEqual(
    lossMask([7, 8, 9, 0], 2, 2),
    [-100, -100, 9, 0, -100, -100],
  );
  assert.throws(() => lossMask([7], 1));
});
