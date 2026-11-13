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

Label invented vectors, token IDs, probabilities, and fixed-answer demos as **illustrative**. Do not present them as the output of a trained model. Check code snippets by running them, and keep external model examples compatible with their documented APIs. Use `lessonStatus: full` only when the lesson meets this standard.

## Sources and rights

This is an independent companion to *Hands-On Large Language Models* by Jay Alammar and Maarten Grootendorst. Write original explanations and make original diagrams. Link to the book’s [official notebooks](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models) and primary papers or documentation for technical claims. Do not add the book PDF, scans, figures, or long copied passages to the repository. The project’s MIT license covers original repository work, not the book.

## Review checklist

- Follow the lesson as a beginner: can you explain the example without opening the book?
- Check all links, including graph connections and source links.
- Test keyboard controls and the text alternative for a new visual or simulation.
- Check light and dark themes, narrow screens, and reduced-motion behavior.
- Run `npm run release:build`; fix the cause of any failed check.

For corrections, please describe what a reader could misunderstand and how your change resolves it. For new concepts, mention the book chapter and the intended parent and prerequisites in the pull request.
