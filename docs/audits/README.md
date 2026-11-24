# E0: source inventory and initial lesson audit

**Completed:** 2026-11-19. This is an initial source and depth audit. Independent technical/editorial acceptance and model execution verification remain later enhancement work.

Read the generated [book gap report](./book-gap-report.md) for the chapter work list, priorities, proposed destinations, and required coverage dimensions. Read the [lesson depth report](./lesson-depth-report.md) for findings on each of the 96 existing concept pages and the two real-model use-case sketches. The [enhancement plan](../../enhancement.md) defines the next steps.

## Sources and method

The source is the supplied first edition of _Hands-On Large Language Models_ (September 2024). The inventory identifies the edition and records the PDF hash without publishing the PDF or the user's local path. All 170 contents headings were matched against normalized text within their recorded PDF page ranges; each entry retains its first matching source page. Detailed contents entries are supplemented with concepts found in body headings, prose, notebook stages, and figures. Printed chapter page `p` maps to PDF viewer page `p + 22`; front matter and nontechnical back matter are treated separately. A section range may include the next heading's page because two sections can share a page.

The [official notebooks](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/tree/ea3390819997999a51983677b80b3aac4dc50ada) were inspected at commit `ea3390819997999a51983677b80b3aac4dc50ada`. The notebook manifest records each blob/hash, its URL, and reviewed cell ranges. Those are zero-based notebook JSON indices. The notebooks were read, not executed; no model training or hosted API was invoked. Original notebooks and book-page renders remain outside the repository.

All concept mechanisms and Python snippets were inspected for their relationship to the topic being taught. Shared code was assessed for purpose rather than automatically treated as a defect. The existing real tokenizer and two real-model use-case sketches are credited, while their unpinned environments, missing run records, and evaluation gaps remain explicit. Twelve representative PDF pages were visually inspected; this is not a claim that every book illustration received a pixel-level review.

## Canonical data

| File                                                            | Responsibility                                                                                                                                          |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [book-sections.json](../../src/data/book-sections.json)         | Edition/source locations, parent relationships, objectives, mappings, dimension requirements/statuses, evidence, gaps, and next phases.                 |
| [lesson-audit.json](../../src/data/lesson-audit.json)           | Per-lesson findings, source hashes, evidence anchors, shared snippets, source-section backreferences, and required work.                                |
| [proposed-concepts.json](../../src/data/proposed-concepts.json) | Proposed standalone pages, subsections, or shared labs; aliases, intended existing parent, and source evidence. These are not implemented destinations. |
| [book-notebooks.json](../../src/data/book-notebooks.json)       | Pinned official notebook provenance, reviewed cells, and differences that must be considered during implementation.                                     |

E9 migrated chapter coverage displays to `book-sections.json` leaf objectives. `book-map.ts` now holds only part/chapter structure; the former topic mapping is archived in `legacy-topic-coverage.json` and is not used at runtime. Canonical concept IDs and lesson URLs remain stable. The separate E9 progress migration preserves existing reading checkmarks and retains the original browser key as a backup.

## Counting and status rules

- Count only `coverageUnit` leaves. A container and its child objectives cannot both count.
- Separate `core` and `context` scope. Research pointers, summaries, publishing matter, and notebook-only bonuses do not inflate book-core coverage.
- `mappingStatus` records whether useful source-objective content has a destination. `partial` does not certify teaching completeness.
- `reviewStatus` is independent of mapping. Legacy `lessonStatus: full` records the old content format; it is not enhancement acceptance.
- `realModelExecution` is independent of both. Running the legacy standard-library snippet checker does not verify a model workflow.
- Every dimension marked not applicable has a rationale. A short contextual discussion need not acquire an artificial training notebook; a training workflow does need real execution and evaluation.
- P0 marks a code/teaching mismatch, P1 a missing core workflow or concept, and P2 further depth for an introductory explanation. P0 is an educational priority, not a security severity.
- The initial audit does not award accepted enhancement coverage. That is deliberate: source mapping and a structural checklist do not establish the stricter teaching rubric.

## Maintain the audit

```bash
npm run test:coverage
npm run coverage:report
npm run check:coverage
npm run release:build
```

The report command validates canonical data and regenerates both Markdown reports. The check command also rejects stale reports. The release build includes coverage validation and mutation tests, then the existing snippet, Astro, graph, and built-link checks. GPU training is not part of this gate.

When editing a lesson, reassess its finding, remaining work, and source mappings before updating its source hash. Do not automatically refresh hashes to bypass a stale-review error. When adding a lesson, register its audit and source mappings; resolve any related proposal rather than leaving a future destination mislabeled as implemented. A later accepted review needs explicit acceptance evidence, completed required dimensions, and execution evidence for required model workflows. E2 will add the fuller review UI and example manifest from the enhancement plan.

The validator checks IDs, chapter/page ranges, the PDF offset, parent cycles, leaf counting, source anchors, mapping backreferences, proposal aliases/parents, source hashes, notebook provenance/cell ranges, applicability reasons, acceptance evidence, and generated-report drift. It cannot prove an explanation is correct or understandable: human and technical review remain necessary.

## First work after E0

Implement **E1**, the reproducible example infrastructure. Then complete the **E2 tokenization, LoRA, and NER reference lessons**. Use the per-topic findings to replace unrelated calculations, add real workflows, and validate actual learning objectives. AP/MAP, source chunking, and other embedded book topics discovered by this audit are now explicit inventory entries.

## E2 update

The three reference lessons now have [author review evidence](../reviews/e2-reference-lessons.md), recorded CPU workflows, and separate review frontmatter. E0 audit dates remain historical; generated reports show the latest enhancement update. One proposed tokenizer-comparison subsection now maps to implemented content. Independent enhancement acceptance remains pending.
