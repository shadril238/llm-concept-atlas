# Contributing to LLM Concept Atlas

Thank you for improving the atlas. The most useful contributions help a beginner understand **what a concept is, how it works, when to use it, and where the simple explanation stops working**.

## Start locally

Use a recent Node.js version, then run:

```bash
npm ci
npm run dev
```

The site uses the `/llm-concept-atlas/` base path. Run `npm run release:build` before opening a pull request. That command checks the concept graph, lesson completeness, runnable Python examples, Astro types, the production build, and internal links.

## Add or improve a concept lesson

Concept pages live in `src/content/docs/concepts/`. Each page has a stable `conceptId` and frontmatter for `chapter`, `part`, `parent`, `prerequisites`, and `related`. The [implementation plan](./plan.md) explains how those fields generate the atlas graph. A public concept must have a destination page and one parent; keep prerequisite links free of cycles.

A complete lesson should include:

1. A precise definition and a plain-language intuition.
2. Linked prerequisites and a step-by-step mechanism.
3. One worked example with intermediate results.
4. An original labeled visual; add a simulation when changing an input teaches the mechanism.
5. Runnable code with setup instructions, expected behavior, and an explanation of the important lines.
6. Use cases, decisions, limits, common misconceptions, and at least two self-checks with answers.
7. The relevant book chapter and the authors’ notebook; add a primary paper or official documentation when it supports a technical detail.

Label invented vectors, token IDs, probabilities, and fixed-answer demos as **illustrative**. Do not present them as the output of a trained model. Check code snippets by running them, and keep external model examples compatible with their documented APIs. The existing `lessonStatus: full` value describes the original lesson format; it does not certify enhancement review or book completeness.

Follow [enhancement.md](./enhancement.md) for the deeper teaching standard and the [audit guide](./docs/audits/README.md) for source mapping, per-lesson findings, and report maintenance. When changing a lesson, reassess its audit before updating the source hash. Run `npm run coverage:report` after reviewed data changes; the release gate rejects broken mappings, stale audits, and stale reports. Keep proposed destinations separate from implemented lessons and real-model execution evidence separate from structural checks.

## Sources and rights

This is an independent companion to *Hands-On Large Language Models* by Jay Alammar and Maarten Grootendorst. Write original explanations and make original diagrams. Link to the book’s [official notebooks](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models) and primary papers or documentation for technical claims. Do not add the book PDF, scans, figures, or long copied passages to the repository. The project’s MIT license covers original repository work, not the book.

## Review checklist

- Follow the lesson as a beginner: can you explain the example without opening the book?
- Check all links, including graph connections and source links.
- Test keyboard controls and the text alternative for a new visual or simulation.
- Check light and dark themes, narrow screens, and reduced-motion behavior.
- Run `npm run release:build`; fix the cause of any failed check.

For corrections, please describe what a reader could misunderstand and how your change resolves it. For new concepts, mention the book chapter and the intended parent and prerequisites in the pull request.

## Example infrastructure

Follow [examples/README.md](./examples/README.md). Register every Python block, use named source regions for new examples, run the declared locked environment, and regenerate synchronized lessons/notebooks. Ordinary site builds execute the foundation suite; CPU model runs are explicit. Preserve the legacy exact-output validations while migrating examples. Passing execution does not resolve an E0 teaching finding or award editorial acceptance.

Use the [reference lesson template](./docs/templates/reference-lesson.md) and [E2 author review](./docs/reviews/e2-reference-lessons.md) when expanding lessons. Keep author review, independent editorial acceptance, and execution verification distinct.

The [E3 packet](./docs/reviews/e3-foundations.md) adds shared foundation mechanisms and source-to-browser parity tests. When changing a numeric explorer, update/test the Python and browser contracts together, and rerun only affected model workflows before refreshing evidence. Aliases belong to canonical pages; relationship reasons must describe actual prerequisite/related/parent links.

E4 data and model changes must retain [UCI attribution and data licenses](./examples/data/manifests/sentiment.json), preserve split/label/invalid-output contracts, and rerun the [discovery workflows](./examples/applications/discovery/README.md). Representation-only topic comparisons must assert unchanged assignments; a topic-label change is not clustering improvement.
