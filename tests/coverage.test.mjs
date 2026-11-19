import test from "node:test";
import assert from "node:assert/strict";
import {
  loadCoverage,
  renderCoverageReports,
  validateCoverage,
} from "../scripts/lib/coverage.mjs";

const baseline = loadCoverage();
const changed = (change) => {
  const data = structuredClone(baseline);
  change(data);
  return validateCoverage(data);
};

test("source inventory, every existing lesson, and pinned notebook references validate", () => {
  assert.deepEqual(validateCoverage(baseline), []);
});

test("broken lesson anchors and unknown proposed destinations are rejected", () => {
  const errors = changed(({ inventory }) => {
    const section = inventory.sections.find((s) => s.mappings.length);
    section.mappings[0].anchor = "not-an-existing-heading";
    section.plannedDestinations.push("unregistered-proposal");
  });
  assert.ok(errors.some((error) => error.includes("missing anchor")));
  assert.ok(errors.some((error) => error.includes("unknown proposal")));
});

test("a changed lesson cannot silently retain its old review", () => {
  assert.ok(
    changed(({ audit }) => {
      audit.lessons[0].sourceSha256 = "0".repeat(64);
    }).some((error) => error.includes("stale lesson audit")),
  );
});

test("omitted lesson audits and duplicate source identifiers fail", () => {
  const errors = changed(({ audit, inventory }) => {
    audit.lessons.pop();
    inventory.sections.push(inventory.sections[0]);
  });
  assert.ok(errors.some((error) => error.includes("Missing lesson audit")));
  assert.ok(errors.some((error) => error.includes("Duplicate section")));
});

test("printed/PDF page confusion and invalid notebook cell references fail", () => {
  const errors = changed(({ inventory, notebooks }) => {
    inventory.sections[0].pdfPages[0] = inventory.sections[0].printedPages[0];
    notebooks.notebooks[0].reviewedCellRanges.push([0, 9999]);
  });
  assert.ok(errors.some((error) => error.includes("PDF page mapping")));
  assert.ok(errors.some((error) => error.includes("invalid cell range")));
});

test("every contents entry retains its checked source-heading location", () => {
  const errors = changed(({ inventory }) => {
    const section = inventory.sections.find((s) => s.origin === "contents");
    delete section.sourceLocationCheck;
  });
  assert.ok(
    errors.some((error) => error.includes("source heading-location evidence")),
  );
});

test("source containers cannot inflate the denominator and parent cycles fail", () => {
  const errors = changed(({ inventory }) => {
    const parent = inventory.sections.find((s) => s.kind === "container");
    parent.coverageUnit = true;
    parent.parentId = parent.id;
  });
  assert.ok(errors.some((error) => error.includes("denominator double count")));
  assert.ok(errors.some((error) => error.includes("parent cycle")));
});

test("unsupported acceptance and unexplained not-applicable dimensions fail", () => {
  const errors = changed(({ inventory }) => {
    const section = inventory.sections.find(
      (s) => s.kind === "workflow" && s.coverageUnit,
    );
    section.reviewStatus = "accepted";
    const context = inventory.sections.find(
      (s) => s.scope === "context" && s.coverageUnit,
    );
    delete context.notApplicableReasons.implementation;
  });
  assert.ok(
    errors.some((error) => error.includes("accepted without completed")),
  );
  assert.ok(errors.some((error) => error.includes("without execution record")));
  assert.ok(errors.some((error) => error.includes("not-applicable reason")));
});

test("proposal alias collisions and broken reverse mappings fail", () => {
  const errors = changed(({ proposals, audit }) => {
    proposals.proposals[0].aliases = ["duplicate alias"];
    proposals.proposals[1].aliases = ["Duplicate Alias"];
    audit.lessons[0].sourceSectionIds.push("book-ch01-not-present");
  });
  assert.ok(errors.some((error) => error.includes("Duplicate proposed alias")));
  assert.ok(errors.some((error) => error.includes("broken reverse mapping")));
});

test("reports deterministically include every source objective and lesson finding", () => {
  const reports = renderCoverageReports(baseline);
  assert.deepEqual(reports, renderCoverageReports(baseline));
  for (const section of baseline.inventory.sections)
    assert.ok(
      reports["docs/audits/book-gap-report.md"].includes(`**${section.id}**`),
    );
  for (const lesson of baseline.audit.lessons)
    assert.ok(
      reports["docs/audits/lesson-depth-report.md"].includes(lesson.finding),
    );
});

test("implemented source destinations require real mappings and review evidence", () => {
  const errors = changed(({ proposals }) => {
    const destination = proposals.proposals.find(p => p.status === 'implemented');
    destination.destination.anchor = 'missing-reference-anchor';
    destination.reviewRecord = null;
  });
  assert.ok(errors.some(error => error.includes('invalid source proposal link')));
  assert.ok(errors.some(error => error.includes('lacks evidence')));
});

test("reference lessons cannot silently lose their author review record", () => {
  const errors = changed(({ audit }) => {
    audit.lessons.find(lesson => lesson.enhancementReview).enhancementReview = 'missing-review';
  });
  assert.ok(errors.some(error => error.includes('missing enhancement review record')));
});
