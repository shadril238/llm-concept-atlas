import { createHash } from "node:crypto";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { resolve } from "node:path";

export const dimensions = [
  "explanation",
  "workedExample",
  "visual",
  "implementation",
  "evaluation",
  "exercises",
];
export const sha256 = (text) => createHash("sha256").update(text).digest("hex");

export function loadCoverage(root = process.cwd()) {
  const read = (name) =>
    JSON.parse(readFileSync(resolve(root, `src/data/${name}.json`), "utf8"));
  return {
    inventory: read("book-sections"),
    audit: read("lesson-audit"),
    proposals: read("proposed-concepts"),
    notebooks: read("book-notebooks"),
  };
}

// The inventory currently references plain ASCII MDX headings. Explicit IDs and
// more complex headings should be resolved through the built HTML when introduced.
export function headingAnchors(source) {
  const seen = new Map();
  return new Set(
    [...source.matchAll(/^#{1,6} (.+)$/gm)].map((match) => {
      const stem = match[1]
        .toLowerCase()
        .replace(/[^\w\- ]/g, "")
        .replace(/ /g, "-");
      const count = seen.get(stem) ?? 0;
      seen.set(stem, count + 1);
      return count ? `${stem}-${count}` : stem;
    }),
  );
}

export function validateCoverage(data, root = process.cwd()) {
  const { inventory, audit, proposals, notebooks } = data;
  const errors = [];
  const check = (value, message) => {
    if (!value) errors.push(message);
  };
  const unique = (items, key, label) => {
    const map = new Map();
    for (const item of items) {
      check(!map.has(item[key]), `Duplicate ${label}: ${item[key]}`);
      map.set(item[key], item);
    }
    return map;
  };
  const sections = unique(inventory.sections, "id", "section");
  const lessons = unique(audit.lessons, "conceptId", "lesson audit");
  const planned = unique(proposals.proposals, "id", "proposal");
  const chapters = unique(inventory.chapters, "number", "chapter");
  const notebookByChapter = unique(notebooks.notebooks, "chapter", "notebook");
  const datePattern = /^\d{4}-\d{2}-\d{2}$/;
  check(datePattern.test(inventory.auditDate), "Invalid audit date");
  check(
    inventory.review.independentReview === false,
    "E0 must not claim independent acceptance",
  );
  check(
    inventory.schemaVersion === 1 && audit.schemaVersion === 1,
    "Unsupported coverage schema version",
  );
  check(
    /^[a-f0-9]{40}$/.test(notebooks.commit),
    "Notebook snapshot must pin a commit",
  );
  check(
    notebooks.execution === "not-run",
    "Static notebook review must not claim execution",
  );
  const conceptFiles = readdirSync(
    resolve(root, "src/content/docs/concepts"),
  ).filter((f) => f.endsWith(".mdx"));
  for (const file of conceptFiles)
    check(lessons.has(file.slice(0, -4)), `Missing lesson audit: ${file}`);
  check(
    lessons.size === conceptFiles.length,
    "Lesson audit contains missing or extra concept files",
  );
  const cache = new Map();
  const source = (path) => {
    if (cache.has(path)) return cache.get(path);
    if (
      !/^src\/content\/docs\/[a-z0-9/-]+\.mdx$/.test(path ?? "") ||
      !existsSync(resolve(root, path))
    ) {
      errors.push(`Invalid source path: ${path}`);
      return "";
    }
    const text = readFileSync(resolve(root, path), "utf8");
    cache.set(path, text);
    return text;
  };
  const link = (entry, label) => {
    const text = source(entry.path);
    check(
      headingAnchors(text).has(entry.anchor),
      `${label}: missing anchor ${entry.path}#${entry.anchor}`,
    );
  };
  const validRange = (range) =>
    Array.isArray(range) &&
    range.length === 2 &&
    range.every(Number.isInteger) &&
    range[0] > 0 &&
    range[1] >= range[0];
  for (let chapter = 1; chapter <= 12; chapter++) {
    check(chapters.has(chapter), `Missing chapter ${chapter}`);
    check(notebookByChapter.has(chapter), `Missing notebook ${chapter}`);
    check(
      inventory.sections.some(
        (s) => s.chapter === chapter && s.origin === "contents",
      ),
      `Chapter ${chapter} lacks contents entries`,
    );
    check(
      inventory.sections.some((s) => s.chapter === chapter && s.coverageUnit),
      `Chapter ${chapter} lacks leaf objectives`,
    );
  }
  check(
    chapters.size === 12 && notebookByChapter.size === 12,
    "Expected the twelve book chapters",
  );
  for (const chapter of chapters.values()) {
    check(
      validRange(chapter.printedPages) && validRange(chapter.pdfPages),
      `Chapter ${chapter.number}: invalid range`,
    );
    check(
      chapter.printedPages.every(
        (p, i) =>
          chapter.pdfPages[i] === p + inventory.source.printedToPdfOffset,
      ),
      `Chapter ${chapter.number}: incorrect PDF offset`,
    );
  }
  const parentIds = new Set(
    inventory.sections.map((s) => s.parentId).filter(Boolean),
  );
  for (const section of sections.values()) {
    const label = section.id;
    const chapter = chapters.get(section.chapter);
    check(
      chapter &&
        label.startsWith(`book-ch${String(section.chapter).padStart(2, "0")}-`),
      `${label}: invalid chapter or ID`,
    );
    check(
      ["core", "context", "prerequisite", "extension"].includes(section.scope),
      `${label}: invalid scope`,
    );
    check(
      ["concept", "workflow", "context", "container"].includes(section.kind),
      `${label}: invalid kind`,
    );
    check(
      ["contents", "body-concept"].includes(section.origin),
      `${label}: invalid source origin`,
    );
    check(
      validRange(section.printedPages) && validRange(section.pdfPages),
      `${label}: invalid page ranges`,
    );
    if (chapter)
      check(
        section.printedPages[0] >= chapter.printedPages[0] &&
          section.printedPages[1] <= chapter.printedPages[1],
        `${label}: outside chapter`,
      );
    check(
      section.printedPages.every(
        (p, i) =>
          section.pdfPages[i] === p + inventory.source.printedToPdfOffset,
      ),
      `${label}: incorrect PDF page mapping`,
    );
    check(
      section.pdfPages[1] <= inventory.source.pdfPageCount,
      `${label}: outside PDF`,
    );
    if (section.origin === "contents") {
      const evidence = section.sourceLocationCheck;
      check(
        evidence?.method === "normalized-heading-match-in-source-page" &&
          evidence.pdfPage >= section.pdfPages[0] &&
          evidence.pdfPage <= section.pdfPages[1],
        `${label}: missing or invalid source heading-location evidence`,
      );
    }
    if (section.parentId) {
      const parent = sections.get(section.parentId);
      check(
        parent && parent.chapter === section.chapter,
        `${label}: missing or cross-chapter parent`,
      );
      if (parent)
        check(
          section.printedPages[0] >= parent.printedPages[0] &&
            section.printedPages[1] <= parent.printedPages[1],
          `${label}: outside parent page range`,
        );
    }
    check(
      section.coverageUnit === !parentIds.has(label),
      `${label}: parent/child denominator double count`,
    );
    check(
      (section.kind === "container") === parentIds.has(label),
      `${label}: inconsistent container kind`,
    );
    check(
      section.objective?.trim() && section.gaps?.length,
      `${label}: missing objective or gap`,
    );
    check(
      section.reviewRecord === inventory.review.id,
      `${label}: unknown review record`,
    );
    check(
      ["P0", "P1", "P2"].includes(section.priority),
      `${label}: invalid priority`,
    );
    check(
      /^E(?:[0-9]|10)$/.test(section.nextPhase),
      `${label}: invalid next phase`,
    );
    check(
      section.plannedLabs.every((id) => /^L\d{2}$/.test(id)),
      `${label}: invalid planned lab`,
    );
    check(
      ["unmapped", "partial", "mapped"].includes(section.mappingStatus),
      `${label}: invalid mapping status`,
    );
    check(
      ["unreviewed", "changes-required", "accepted"].includes(
        section.reviewStatus,
      ),
      `${label}: invalid review status`,
    );
    check(
      section.mappingStatus !== "unmapped" || section.mappings.length === 0,
      `${label}: unmapped section has destinations`,
    );
    check(
      section.mappingStatus === "unmapped" || section.mappings.length > 0,
      `${label}: mapped/partial section lacks destination`,
    );
    check(
      section.mappings.length || section.plannedDestinations.length,
      `${label}: no existing or planned destination`,
    );
    for (const entry of section.mappings) {
      link(entry, label);
      if (entry.conceptId)
        check(
          lessons.has(entry.conceptId) &&
            entry.path === lessons.get(entry.conceptId).path,
          `${label}: unknown or mismatched concept ${entry.conceptId}`,
        );
    }
    for (const evidence of section.evidence) {
      link(evidence, label);
      check(evidence.finding?.trim(), `${label}: empty evidence`);
    }
    for (const id of section.plannedDestinations)
      check(planned.has(id), `${label}: unknown proposal ${id}`);
    for (const dimension of dimensions) {
      const required = section.requirements[dimension],
        status = section.coverage[dimension];
      check(
        typeof required === "boolean",
        `${label}: missing requirement ${dimension}`,
      );
      check(
        ["missing", "partial", "sufficient", "not-applicable"].includes(status),
        `${label}: invalid ${dimension} status`,
      );
      check(
        required ? status !== "not-applicable" : status === "not-applicable",
        `${label}: inconsistent applicability for ${dimension}`,
      );
      if (!required)
        check(
          section.notApplicableReasons[dimension]?.trim(),
          `${label}: no not-applicable reason for ${dimension}`,
        );
      if (required && status === "sufficient")
        check(
          section.evidence.length > 0,
          `${label}: sufficient ${dimension} lacks evidence`,
        );
    }
    if (section.reviewStatus === "accepted") {
      check(
        section.mappingStatus === "mapped" &&
          dimensions.every(
            (d) =>
              !section.requirements[d] || section.coverage[d] === "sufficient",
          ),
        `${label}: accepted without completed required dimensions`,
      );
      check(
        section.acceptanceRecord?.reviewer && section.acceptanceRecord?.date,
        `${label}: missing acceptance evidence`,
      );
      if (section.requirements.implementation)
        check(
          section.realModelExecution === "verified" && section.executionRecord,
          `${label}: accepted implementation without execution record`,
        );
    }
    const visited = new Set([label]);
    let next = section.parentId;
    while (next && sections.has(next)) {
      if (visited.has(next)) {
        errors.push(`${label}: source parent cycle`);
        break;
      }
      visited.add(next);
      next = sections.get(next).parentId;
    }
  }
  for (const lesson of lessons.values()) {
    const text = source(lesson.path);
    check(
      sha256(text) === lesson.sourceSha256,
      `${lesson.conceptId}: stale lesson audit; reassess the changed source`,
    );
    check(
      new RegExp(`^conceptId: ${lesson.conceptId}$`, "m").test(text),
      `${lesson.conceptId}: source ID mismatch`,
    );
    check(
      new RegExp(`^chapter: ${lesson.chapter}$`, "m").test(text),
      `${lesson.conceptId}: source chapter mismatch`,
    );
    link(
      { path: lesson.path, anchor: lesson.evidenceAnchor },
      lesson.conceptId,
    );
    check(
      lesson.finding?.trim() &&
        lesson.requiredWork.length > 0 &&
        lesson.sourceSectionIds.length > 0,
      `${lesson.conceptId}: incomplete audit`,
    );
    check(
      lesson.reviewRecord === inventory.review.id,
      `${lesson.conceptId}: unknown audit record`,
    );
    check(
      lesson.reviewStatus === "changes-required",
      `${lesson.conceptId}: E0 audit is not teaching acceptance`,
    );
    for (const id of lesson.sourceSectionIds)
      check(
        sections
          .get(id)
          ?.mappings.some((m) => m.conceptId === lesson.conceptId),
        `${lesson.conceptId}: broken reverse mapping ${id}`,
      );
    for (const id of lesson.sharedExampleWith)
      check(
        lessons.has(id) && lessons.get(id).codeSha256 === lesson.codeSha256,
        `${lesson.conceptId}: invalid shared-code reference ${id}`,
      );
  }
  const reviewRecords = JSON.parse(
    readFileSync(resolve(root, "src/data/review-records.json"), "utf8"),
  ).records;
  const examples = JSON.parse(
    readFileSync(resolve(root, "examples/manifest.json"), "utf8"),
  ).examples;
  for (const lesson of lessons.values()) {
    if (!lesson.enhancementReview) continue;
    const review = reviewRecords.find(
      (record) => record.id === lesson.enhancementReview,
    );
    check(
      review?.concepts.includes(lesson.conceptId),
      `${lesson.conceptId}: missing enhancement review record`,
    );
    check(
      review?.independentReview === false && review?.accepted === false,
      `${lesson.conceptId}: author review must not claim independent acceptance`,
    );
    const text = source(lesson.path);
    for (const field of [
      "bookSections",
      "exampleIds",
      "learningObjectives",
      "aliases",
    ]) {
      const match = text.match(new RegExp(`^${field}: (.+)$`, "m"));
      const values = match ? JSON.parse(match[1]) : [];
      check(
        Array.isArray(values) && values.length > 0,
        `${lesson.conceptId}: missing ${field}`,
      );
      if (field === "bookSections")
        check(
          values.every(
            (id) => sections.has(id) && lesson.sourceSectionIds.includes(id),
          ),
          `${lesson.conceptId}: invalid lesson source sections`,
        );
      if (field === "exampleIds")
        check(
          values.every((id) =>
            examples.some(
              (example) =>
                example.id === id &&
                example.concepts.includes(lesson.conceptId),
            ),
          ),
          `${lesson.conceptId}: invalid lesson example IDs`,
        );
    }
    check(
      text.includes(`reviewRecord: ${lesson.enhancementReview}`),
      `${lesson.conceptId}: review frontmatter mismatch`,
    );
  }
  for (const useCase of audit.useCases) {
    link(useCase, "use case");
    check(
      sha256(source(useCase.path)) === useCase.sourceSha256,
      `${useCase.path}: stale use-case audit`,
    );
  }
  const aliases = new Map();
  for (const proposal of planned.values()) {
    check(
      !lessons.has(proposal.id),
      `${proposal.id}: proposal duplicates a canonical concept`,
    );
    check(
      lessons.has(proposal.parentConceptId),
      `${proposal.id}: invalid planned parent`,
    );
    check(
      ["standalone", "subsection", "lab"].includes(proposal.placement),
      `${proposal.id}: invalid placement`,
    );
    check(
      ["proposed", "implemented"].includes(proposal.status),
      `${proposal.id}: invalid destination status`,
    );
    check(
      proposal.sourceSectionIds.length > 0,
      `${proposal.id}: proposal lacks source evidence`,
    );
    for (const id of proposal.sourceSectionIds)
      check(
        proposal.status === "implemented"
          ? sections
              .get(id)
              ?.mappings.some(
                (m) =>
                  m.path === proposal.destination?.path &&
                  m.anchor === proposal.destination?.anchor,
              )
          : sections.get(id)?.plannedDestinations.includes(proposal.id),
        `${proposal.id}: invalid source proposal link ${id}`,
      );
    if (proposal.status === "implemented") {
      link(proposal.destination, proposal.id);
      check(
        proposal.reviewRecord && proposal.exampleIds?.length,
        `${proposal.id}: implemented destination lacks evidence`,
      );
    }
    for (const alias of proposal.aliases) {
      const normalized = alias.toLowerCase().trim();
      check(!aliases.has(normalized), `Duplicate proposed alias: ${alias}`);
      aliases.set(normalized, proposal.id);
    }
  }
  for (const notebook of notebookByChapter.values()) {
    check(
      /^[a-f0-9]{64}$/.test(notebook.sha256) &&
        /^[a-f0-9]{40}$/.test(notebook.blob),
      `Notebook ${notebook.chapter}: missing source hashes`,
    );
    check(
      notebook.url.includes(`/blob/${notebooks.commit}/`),
      `Notebook ${notebook.chapter}: unpinned URL`,
    );
    for (const [first, last] of notebook.reviewedCellRanges)
      check(
        first >= 0 && last >= first && last < notebook.cells,
        `Notebook ${notebook.chapter}: invalid cell range`,
      );
  }
  return errors;
}

const cell = (text) => String(text).replace(/\|/g, "\\|").replace(/\n/g, " ");
const docLink = (entry, label = entry.conceptId ?? "use-case sketch") =>
  `[${cell(label)}](../../${entry.path}#${entry.anchor})`;

export function renderCoverageReports({
  inventory,
  audit,
  proposals,
  notebooks,
}) {
  const leaves = inventory.sections.filter((s) => s.coverageUnit);
  const counts = (list, predicate) => list.filter(predicate).length;
  const gap = [
    "# Book coverage and enhancement gap report",
    "",
    `Generated from the E0 inventory dated ${inventory.auditDate}; latest enhancement update ${inventory.lastUpdated ?? inventory.auditDate}. Rebuild with \`npm run coverage:report\`; do not edit generated tables directly.`,
    "",
    "**This is a source and depth audit, not a certification of complete teaching coverage.** All legacy lessons require enhancement review. Existing mappings are partial; an official notebook link is a source, not an atlas implementation.",
    "",
    `The inventory records **${inventory.sections.length} source entries** (${counts(inventory.sections, (s) => s.origin === "contents")} contents entries and ${counts(inventory.sections, (s) => s.origin === "body-concept")} embedded concepts). It counts **${leaves.length} leaf objectives**; ${inventory.sections.length - leaves.length} parent containers are excluded from the denominator. This inventory is a reviewed edition-specific work list, not a universal count of all AI terms.`,
    "",
    `Of the leaf objectives, **${counts(leaves, (s) => s.mappingStatus === "partial")} have partial atlas mappings** and **${counts(leaves, (s) => s.mappingStatus === "unmapped")} have no direct atlas mapping**. **${counts(leaves, (s) => s.reviewStatus === "accepted")} have passed enhancement acceptance**. Planned destinations are not published lessons.`,
    "",
    "## Read this report",
    "",
    "- **P0:** a code/example mismatch that risks teaching the wrong mechanism; correct before expanding that topic.",
    "- **P1:** missing core explanation or real workflow; implement in its planned enhancement phase.",
    "- **P2:** deepen an existing introductory mechanism, trace, visual, or exercise.",
    "- **Partial** credits useful introductory material; it does not mean a full workflow is available.",
    "- Source page ranges are printed pages. PDF viewer pages are explicitly stored in the inventory; overlapping boundary pages can contain two section headings.",
    "- `implementation` means an actual model/library workflow when required. Contextual and mechanism-only objectives have explicit not-applicable reasons; their worked examples still matter.",
    "",
    "## Chapter summary",
    "",
    "| Chapter | Source entries | Core leaves | Context leaves | Partial mappings | Unmapped | Accepted | Next batch |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
  ];
  for (const chapter of inventory.chapters) {
    const all = inventory.sections.filter((s) => s.chapter === chapter.number),
      units = all.filter((s) => s.coverageUnit);
    gap.push(
      `| ${chapter.number}. ${chapter.title} | ${all.length} | ${counts(units, (s) => s.scope === "core")} | ${counts(units, (s) => s.scope === "context")} | ${counts(units, (s) => s.mappingStatus === "partial")} | ${counts(units, (s) => s.mappingStatus === "unmapped")} | ${counts(units, (s) => s.reviewStatus === "accepted")} | ${chapter.nextPhase} |`,
    );
  }
  gap.push(
    "",
    "## Required dimensions",
    "",
    "| Dimension | Required leaf objectives | Missing | Partial | Sufficient |",
    "| --- | ---: | ---: | ---: | ---: |",
  );
  for (const d of dimensions) {
    const required = leaves.filter((s) => s.requirements[d]);
    gap.push(
      `| ${d} | ${required.length} | ${counts(required, (s) => s.coverage[d] === "missing")} | ${counts(required, (s) => s.coverage[d] === "partial")} | ${counts(required, (s) => s.coverage[d] === "sufficient")} |`,
    );
  }
  gap.push(
    "",
    "## Priority corrections",
    "",
    "| Existing lesson | Evidence and correction | Next phase |",
    "| --- | --- | --- |",
  );
  for (const lesson of audit.lessons.filter((l) => l.priority === "P0"))
    gap.push(
      `| ${docLink({ ...lesson, anchor: lesson.evidenceAnchor })} | ${cell(lesson.finding)} | ${lesson.nextPhase} |`,
    );
  gap.push(
    "",
    "## Proposed destinations",
    "",
    "The placement decision avoids creating a separate page for every subsection. IDs remain proposals until implemented; aliases below are search vocabulary to add later.",
    "",
    "| Proposed ID | Placement | Existing parent | Aliases | Source references |",
    "| --- | --- | --- | --- | --- |",
  );
  for (const p of proposals.proposals.filter((p) => p.status === "proposed"))
    gap.push(
      `| ${p.id} | ${p.placement} | ${p.parentConceptId} | ${cell(p.aliases.join(", ") || "—")} | ${p.sourceSectionIds.join(", ")} |`,
    );
  for (const chapter of inventory.chapters) {
    const notebook = notebooks.notebooks.find(
      (n) => n.chapter === chapter.number,
    );
    gap.push(
      "",
      `## Chapter ${chapter.number}: ${chapter.title}`,
      "",
      `Printed pages ${chapter.printedPages.join("–")}; PDF pages ${chapter.pdfPages.join("–")}. [Pinned official notebook](${notebook.url}); reviewed cell ranges (zero-based): ${notebook.reviewedCellRanges.map((r) => r.join("–")).join(", ")}. Planned lab packages: ${chapter.plannedLabs.join(", ")}. Notebook execution was not part of E0.`,
      "",
      "| Source ID / location | Objective and remaining work | Existing destination | Status / priority |",
      "| --- | --- | --- | --- |",
    );
    for (const s of inventory.sections.filter(
      (item) => item.chapter === chapter.number,
    )) {
      const target =
        s.mappings.map((m) => docLink(m)).join(", ") ||
        `Proposed: ${s.plannedDestinations.join(", ")}`;
      gap.push(
        `| **${s.id}**<br>${cell(s.title)}; pp. ${s.printedPages.join("–")} (${s.origin}${s.coverageUnit ? "" : "; container, not counted"}) | ${cell(s.objective)} | ${target} | ${s.mappingStatus}; ${s.priority}; ${s.nextPhase} |`,
      );
    }
  }
  gap.push(
    "",
    "## Source-review cautions",
    "",
    ...notebooks.differences.map(
      (item) =>
        `- Chapter ${item.chapter}, notebook cells ${item.cells.join(", ")}: ${item.finding}`,
    ),
    ...inventory.review.sourceCautions.map(
      (item) =>
        `- Chapter ${item.chapter}, printed pages ${item.printedPages.join("–")}: ${item.note}`,
    ),
    "",
    "## Scope and exclusions",
    "",
    ...inventory.sourceExclusions.map(
      (item) => `- **${item.title}:** ${item.treatment}`,
    ),
    "",
    "## Review limits and next action",
    "",
    ...inventory.review.limitations.map((line) => `- ${line}`),
    "",
    "E0 produces the inventory and prioritization. Next implement E1 reproducible example infrastructure, then E2 tokenization, LoRA, and NER reference lessons. Follow [enhancement.md](../../enhancement.md) for the acceptance gates. See the [audit guide](./README.md) for maintenance and evidence rules.",
    "",
  );
  const depth = [
    "# Existing lesson depth audit",
    "",
    `Initial review ${inventory.auditDate}; latest lesson update ${audit.lastUpdated ?? inventory.auditDate}. Covers **${audit.lessons.length} concepts**. Source hashes detect stale findings. Independent pedagogical acceptance remains pending; execution claims below refer only to linked workflow evidence.`,
    "",
    `The ${audit.lessons.length} concept snippets contain **${new Set(audit.lessons.map((l) => l.codeSha256)).size} distinct Python implementations**. Shared code is not automatically wrong; the finding records whether it demonstrates this lesson's mechanism.`,
    "",
    "## Useful existing work",
    "",
    "The site has explanations, labeled illustrative examples, relationship navigation, and browser labs worth retaining. Tokenization includes a real-tokenizer sketch; the two use-case pages contain real-model sketches. These need pinned environments, execution evidence, and fuller evaluation. The deterministic legacy snippet checker is a separate execution check and cannot certify teaching depth.",
    "",
    "## Per-lesson findings",
    "",
  ];
  for (const chapter of inventory.chapters) {
    depth.push(
      `### Chapter ${chapter.number}`,
      "",
      "| Lesson | Current implementation finding | Priority / next phase |",
      "| --- | --- | --- |",
    );
    for (const lesson of audit.lessons.filter(
      (l) => l.chapter === chapter.number,
    ))
      depth.push(
        `| ${docLink({ ...lesson, anchor: lesson.evidenceAnchor })} | ${cell(lesson.finding)} | ${lesson.priority} / ${lesson.nextPhase} |`,
      );
    depth.push("");
  }
  depth.push(
    "## Real-model use-case sketches",
    "",
    ...audit.useCases.map((u) => `- ${docLink(u, u.implementation)}: ${u.gap}`),
    "",
    "## How to act on a finding",
    "",
    "Use the source-section IDs and requiredWork array in `src/data/lesson-audit.json` to identify the exact objectives. Keep a useful illustrative calculation, but do not label it the missing model workflow. Reassess evidence before updating a source hash; do not auto-refresh hashes to silence stale-review failures.",
    "",
  );
  return {
    "docs/audits/book-gap-report.md": gap.join("\n"),
    "docs/audits/lesson-depth-report.md": depth.join("\n"),
  };
}
