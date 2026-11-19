import test from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import {
  attentionRows,
  filteredDistribution,
  ropePair,
  kvBytes,
} from "../src/lib/foundation-mechanics.js";
import { buildAtlasGraph } from "../src/lib/graph.js";

test("browser and Python use identical masked attention and decoding quantities", () => {
  const code = `import json,sys;sys.path.insert(0,'examples/foundations/src');from atlas_foundations.e3.mechanics import attention,filter_distribution,rope_pair;q=[[1.,0.],[0.,1.],[1.,1.]];v=[[1.,0.],[0.,2.],[3.,1.]];print(json.dumps({'weights':attention(q,q,v)[0],'p':filter_distribution([2.,1.,0.],top_p=.8),'rope':rope_pair([1.,0.],2,.5)}))`;
  const run = spawnSync("python3", ["-c", code], { encoding: "utf8" });
  assert.equal(run.status, 0, run.stderr);
  const data = JSON.parse(run.stdout);
  const close = (a, b) => {
    if (Array.isArray(a)) {
      assert.equal(a.length, b.length);
      a.forEach((x, i) => close(x, b[i]));
    } else assert.ok(Math.abs(a - b) < 1e-12);
  };
  close(attentionRows().weights, data.weights);
  close(filteredDistribution([2, 1, 0], 1, 3, 0.8), data.p);
  close(ropePair([1, 0], 2, 0.5), data.rope);
  assert.equal(kvBytes(6, 1, 4, 12, 32, 8), 147456);
  assert.deepEqual(attentionRows("packed").weights[2], [0, 0, 1]);
});
test("aliases and relationship reasons remain in validated graph data", () => {
  const parts = [
    { id: "foundations", shortTitle: "Foundations", summary: "Start" },
  ];
  const chapters = [
    { number: 1, title: "Intro", summary: "Start", part: "foundations" },
  ];
  const base = [
    { id: "parts/foundations", data: {} },
    { id: "chapters/foundations/01", data: {} },
    {
      id: "concepts/test",
      data: {
        kind: "concept",
        conceptId: "test",
        title: "Test",
        summary: "Test",
        chapter: 1,
        part: "foundations",
        parent: "chapter-01",
        aliases: ["example alias"],
        relationshipReasons: { "chapter-01": "Book location" },
      },
    },
  ];
  assert.deepEqual(
    buildAtlasGraph(parts, chapters, base).nodes.at(-1).aliases,
    ["example alias"],
  );
  base.at(-1).data.relationshipReasons = { missing: "invalid" };
  assert.throws(
    () => buildAtlasGraph(parts, chapters, base),
    /reason for unrelated/,
  );
});
