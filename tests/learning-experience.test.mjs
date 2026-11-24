import test from "node:test";
import assert from "node:assert/strict";
import {
  rankRecords,
  normalize,
  suggestions,
  prerequisiteRoute,
  relationshipReason,
} from "../src/lib/discovery.js";
import {
  sanitizeProgress,
  loadProgress,
  reduceProgress,
  legacyKey,
  progressKey,
} from "../src/lib/learning-progress.js";
import { summarizeCoverage } from "../src/lib/coverage-summary.js";
const records = [
  {
    id: "ner",
    title: "Named-entity recognition",
    kind: "concept",
    chapter: 11,
    chapters: [11],
    level: "advanced",
    aliases: ["NER"],
    tasks: ["extract names from text"],
    summary: "Find spans.",
  },
  {
    id: "few",
    title: "Few-shot examples",
    kind: "concept",
    chapter: 6,
    chapters: [6],
    level: "beginner",
    aliases: ["few shot prompting"],
    tasks: [],
    summary: "Give demonstrations.",
  },
  {
    id: "lab",
    title: "Entity lab",
    kind: "lab",
    chapters: [11],
    aliases: [],
    tasks: [],
    summary: "A real workflow.",
  },
];
test("aliases, punctuation, task words and intersected filters behave consistently", () => {
  assert.equal(normalize("c‑TF–IDF"), "c tf idf");
  assert.equal(rankRecords(records, "NER")[0].id, "ner");
  assert.equal(rankRecords(records, "named entity recognition")[0].id, "ner");
  assert.equal(
    rankRecords(records, "extract names from text")[0].match,
    "Task",
  );
  assert.equal(rankRecords(records, "NER", { chapter: "6" }).length, 0);
  assert.equal(
    rankRecords(records, "", { kind: "lab", chapter: "11" }).length,
    1,
  );
  assert.equal(
    rankRecords(records, "", { kind: "lab", level: "advanced" }).length,
    0,
  );
});
test("near-spelling suggestions do not silently invent results", () => {
  assert.equal(rankRecords(records, "NERZ").length, 0);
  assert.ok(suggestions(records, "NERZ").includes("NER"));
  assert.deepEqual(suggestions(records, "unrelatedphantomphrase"), []);
});
test("prerequisites are ordered without duplicates; conceptual parents are not prerequisites", () => {
  const nodes = [
    { id: "root", prerequisites: [] },
    { id: "a", prerequisites: ["root"] },
    { id: "b", prerequisites: ["root"] },
    { id: "c", parent: "other", prerequisites: ["a", "b"] },
  ];
  assert.deepEqual(
    prerequisiteRoute(nodes, "c").map((n) => n.id),
    ["root", "a", "b", "c"],
  );
  assert.throws(
    () => prerequisiteRoute([{ id: "a", prerequisites: ["a"] }], "a"),
    /cycle/,
  );
  assert.throws(() => prerequisiteRoute(nodes, "unknown"), /Unknown/);
});
test("inbound relationships retain their authored explanation", () => {
  assert.equal(
    relationshipReason(
      [
        { id: "a", relationshipReasons: { b: "A compares with B." } },
        { id: "b" },
      ],
      "b",
      "a",
      "related",
    ),
    "A compares with B.",
  );
});
test("progress migrates safely, deduplicates and preserves its legacy backup", () => {
  const memory = new Map([
    [legacyKey, JSON.stringify(["ner", "ner", "obsolete", 42])],
  ]);
  const storage = {
    getItem: (k) => memory.get(k),
    setItem: (k, v) => memory.set(k, v),
  };
  const result = loadProgress(storage, ["ner", "few"], ["route"]);
  assert.deepEqual(result.data.lessons, ["ner"]);
  assert.equal(result.migrated, true);
  assert.ok(memory.has(legacyKey));
  assert.equal(JSON.parse(memory.get(progressKey)).version, 2);
  let next = reduceProgress(
    result.data,
    { type: "practice", id: "route", step: "lab", checked: true },
    ["ner", "few"],
    ["route"],
  );
  assert.equal(next.practice.route.lab, true);
  next = reduceProgress(
    next,
    { type: "lesson", id: "ner" },
    ["ner", "few"],
    ["route"],
  );
  assert.deepEqual(next.lessons, []);
});
test("blocked/malformed storage and unknown practice IDs cannot corrupt progress", () => {
  const blocked = {
    getItem() {
      throw Error("blocked");
    },
    setItem() {
      throw Error("blocked");
    },
  };
  assert.equal(loadProgress(blocked, ["ner"], []).persistent, false);
  const clean = sanitizeProgress(
    {
      lessons: ["unknown", "ner"],
      practice: { unknown: { lab: true } },
      lastConceptId: "javascript:bad",
    },
    ["ner"],
    [],
  );
  assert.deepEqual(clean.lessons, ["ner"]);
  assert.equal(clean.lastConceptId, null);
  assert.deepEqual(clean.practice, {});
});
test("coverage excludes source containers and separates mapping, acceptance and execution", () => {
  const objective = {
    chapter: 12,
    coverageUnit: true,
    mappings: [{}],
    requirements: { implementation: true },
    reviewStatus: "changes-required",
    realModelExecution: "verified",
    executionRecord: "record.json",
  };
  assert.deepEqual(
    summarizeCoverage([objective, { ...objective, coverageUnit: false }]),
    { total: 1, mapped: 1, accepted: 0, practical: 1, executed: 1 },
  );
  assert.equal(summarizeCoverage([objective], 11).total, 0);
});

test("short acronyms do not match fragments of unrelated words", () => {
  const rows = [
    ...records,
    {
      id: "generator",
      title: "Generative model",
      summary: "Produces text.",
      aliases: [],
      tasks: [],
    },
  ];
  assert.ok(!rankRecords(rows, "NER").some((r) => r.id === "generator"));
});
test("a newer progress version is preserved without overwriting its key", () => {
  const value = JSON.stringify({ version: 3, lessons: ["ner"], practice: {} });
  const memory = new Map([[progressKey, value]]);
  const state = loadProgress(
    { getItem: (k) => memory.get(k), setItem: (k, v) => memory.set(k, v) },
    ["ner"],
    [],
  );
  assert.equal(state.protectedStorage, true);
  assert.equal(memory.get(progressKey), value);
});
