import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { summarizeCoverage } from "../src/lib/coverage-summary.js";

// A reproducible inventory of recorded evidence, not a human acceptance decision.
const read = (path) => readFileSync(path, "utf8");
const json = (path) => JSON.parse(read(path));
const inputs = [
  "src/data/book-sections.json",
  "src/data/review-records.json",
  "src/data/book-notebooks.json",
  "examples/manifest.json",
];
const inventory = json(inputs[0]);
const reviews = json(inputs[1]).records;
const notebooks = json(inputs[2]);
const examples = json(inputs[3]).examples;
const leaves = inventory.sections.filter((section) => section.coverageUnit);
const practical = leaves.filter(
  (section) => section.requirements.implementation,
);
const references = new Map();
for (const section of inventory.sections) {
  if (section.realModelExecution !== "verified") continue;
  const path = section.executionRecord;
  if (!path)
    throw new Error(`${section.id}: verified execution lacks evidence`);
  const contents = read(path); // Fail on missing evidence, including indirect author summaries.
  const registered = examples.find(
    (example) => example.verificationRecord === path,
  );
  if (path.endsWith(".json")) {
    const record = JSON.parse(contents);
    if (
      !registered ||
      record.status !== "passed" ||
      record.exampleId !== registered.id
    )
      throw new Error(
        `${section.id}: execution must reference a registered passing record`,
      );
  }
  references.set(path, {
    path,
    sha256: createHash("sha256").update(contents).digest("hex"),
    type: registered ? "execution-record" : "indirect-author-summary",
  });
}
const withRecord = examples.filter((example) => example.verificationRecord);
for (const example of withRecord) {
  const record = json(example.verificationRecord);
  if (record.status !== "passed" || record.exampleId !== example.id)
    throw new Error(`${example.id}: missing passing execution evidence`);
}
const snapshot = {
  schemaVersion: 1,
  purpose:
    "Evidence inventory only. Passing automation and mapped pages do not establish independent teaching acceptance or supported GPU execution.",
  inputs: inputs.map((path) => ({
    path,
    sha256: createHash("sha256").update(read(path)).digest("hex"),
  })),
  coverage: summarizeCoverage(inventory.sections),
  chapters: inventory.chapters.map(({ number, title }) => ({
    number,
    title,
    ...summarizeCoverage(inventory.sections, number),
  })),
  reviews: {
    recorded: reviews.length,
    independent: reviews.filter((review) => review.independentReview === true)
      .length,
    independentlyAccepted: reviews.filter(
      (review) => review.independentReview === true && review.accepted === true,
    ).length,
  },
  examples: {
    registered: examples.length,
    withPassingRecords: withRecord.length,
    unverified: examples
      .filter((example) => !example.verificationRecord)
      .map(({ id }) => id),
  },
  officialNotebookExecution: notebooks.execution,
  unverifiedPracticalObjectives: practical
    .filter(
      (section) =>
        section.realModelExecution !== "verified" || !section.executionRecord,
    )
    .map(({ id, objective, gaps }) => ({ id, objective, gaps })),
  practicalObjectivesWithIndirectEvidence: practical
    .filter(
      (section) =>
        references.get(section.executionRecord)?.type ===
        "indirect-author-summary",
    )
    .map(({ id }) => id),
  executionEvidenceReferences: [...references.values()].sort((a, b) =>
    a.path.localeCompare(b.path),
  ),
};
const output = "docs/reviews/e10-evidence-audit.json";
const content = `${JSON.stringify(snapshot, null, 2)}\n`;
if (process.argv.includes("--write")) writeFileSync(output, content);
else if (read(output) !== content)
  throw new Error(
    `${output} is stale; run npm run evidence:report and review the changes`,
  );
console.log(
  `Evidence inventory: ${snapshot.coverage.executed}/${snapshot.coverage.practical} practical objectives recorded; ${snapshot.coverage.accepted}/${snapshot.coverage.total} independently accepted. This is not a release approval.`,
);
