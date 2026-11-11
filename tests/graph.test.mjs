import test from "node:test";
import assert from "node:assert/strict";
import { buildAtlasGraph } from "../src/lib/graph.js";

const parts = [
  { id: "foundations", shortTitle: "Foundations", summary: "Start here." },
];
const chapters = [
  {
    number: 1,
    title: "Introduction",
    summary: "A start.",
    part: "foundations",
  },
];
const base = [
  { id: "parts/foundations", data: { kind: "part" } },
  { id: "chapters/foundations/01", data: { kind: "chapter" } },
  {
    id: "concepts/token",
    data: {
      kind: "concept",
      conceptId: "token",
      title: "Token",
      summary: "A unit of text.",
      chapter: 1,
      part: "foundations",
      parent: "chapter-01",
      prerequisites: [],
      related: [],
    },
  },
];

test("builds a navigable parent and concept graph", () => {
  const graph = buildAtlasGraph(parts, chapters, base);
  assert.equal(graph.nodes.length, 3);
  assert.deepEqual(
    graph.edges
      .filter((edge) => edge.type === "contains")
      .map(({ source, target }) => [source, target]),
    [
      ["foundations", "chapter-01"],
      ["chapter-01", "token"],
    ],
  );
});

test("rejects a missing related concept", () => {
  const entries = structuredClone(base);
  entries[2].data.related = ["missing"];
  assert.throws(
    () => buildAtlasGraph(parts, chapters, entries),
    /unknown related concept missing/,
  );
});

test("rejects parent cycles", () => {
  const entries = structuredClone(base);
  entries.push({
    id: "concepts/embedding",
    data: {
      kind: "concept",
      conceptId: "embedding",
      title: "Embedding",
      summary: "A vector.",
      chapter: 1,
      part: "foundations",
      parent: "token",
      prerequisites: [],
      related: [],
    },
  });
  entries[2].data.parent = "embedding";
  assert.throws(
    () => buildAtlasGraph(parts, chapters, entries),
    /Parent cycle/,
  );
});

test("rejects prerequisite cycles", () => {
  const entries = structuredClone(base);
  entries.push({
    id: "concepts/embedding",
    data: {
      kind: "concept",
      conceptId: "embedding",
      title: "Embedding",
      summary: "A vector.",
      chapter: 1,
      part: "foundations",
      parent: "chapter-01",
      prerequisites: ["token"],
      related: [],
    },
  });
  entries[2].data.prerequisites = ["embedding"];
  assert.throws(
    () => buildAtlasGraph(parts, chapters, entries),
    /Prerequisite cycle/,
  );
});
