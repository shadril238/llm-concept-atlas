# LLM Concept Atlas

**See it. Try it. Understand it.** An independent visual companion to *Hands-On Large Language Models* by Jay Alammar and Maarten Grootendorst.

The original Phases 1–5 produced a **curated companion across all 12 chapters**: 96 concept lessons, an interactive relationship graph, 11 teaching simulations, three guided paths, and two use cases. The explanations and examples vary in depth; this is not yet exhaustive book coverage or a complete set of evaluated model workflows.

**Enhancements E0 and E1 are complete:** the [book gap report](./docs/audits/book-gap-report.md) inventories 231 source entries and distinguishes partial explanations, missing topics, and practical implementation gaps. The [lesson depth audit](./docs/audits/lesson-depth-report.md) records findings for every existing concept. The [example guide](./examples/README.md) provides locked environments, a 213-example registry, synchronized lesson/notebook source, and recorded CPU verification. Follow [enhancement.md](./enhancement.md) for the remaining enhancement phases; independent teaching acceptance and verification of the remaining model workflows are pending.

**E2 implementation delivered:** [tokenization](https://shadril238.github.io/llm-concept-atlas/concepts/tokenization/), [LoRA](https://shadril238.github.io/llm-concept-atlas/concepts/lora/), and [NER](https://shadril238.github.io/llm-concept-atlas/concepts/named-entity-recognition/) now have detailed reference lessons, interactive traces, real CPU workflows, and topic-specific practice. [Author review and execution evidence](./docs/reviews/e2-reference-lessons.md) are available; independent editorial review and beginner feedback remain pending.

**E3 implementation delivered:** 32 foundation lessons have been rewritten or added, alongside the reused tokenization reference, five CPU workflow notebooks, calculated/recorded experiments, alias-aware graph/directory search, a glossary, and an updated learning path. E3 brought the atlas to 100 concept lessons. See the [foundation labs](https://shadril238.github.io/llm-concept-atlas/labs/foundations/) and [author review](./docs/reviews/e3-foundations.md). Independent editorial/beginner acceptance remains pending.

**E4 implementation delivered:** 15 rewritten/new lessons, seven same-case classification methods, a fitted modular BERTopic workflow, local topic-label generation, and a research collection explorer. See [L07–L08](https://shadril238.github.io/llm-concept-atlas/labs/classification-and-topics/) and the [execution guide](./examples/applications/discovery/README.md). Hosted comparison uses a documented offline importer; no live API result is claimed. Independent review remains pending.

**Read the site:** [llm-concept-atlas on GitHub Pages](https://shadril238.github.io/llm-concept-atlas/).

**E5 implementation delivered:** 18 detailed Chapter 6–7 lessons, including new chain-of-thought/tree-of-thought pages, and five local-model L09–L11 workflows. Explore actual prompt comparisons, native constraints, sampled voting, search branches, memory budgets, and bounded tool traces in [L09–L11](https://shadril238.github.io/llm-concept-atlas/labs/prompting-and-workflows/). Read the [execution guide](./examples/applications/workflows/README.md). Small-model failures remain visible; independent review is pending.

## Run locally

Requires a recent Node.js version.

```bash
npm ci
npm run dev
```

The site uses the `/llm-concept-atlas/` base path. Open the local URL printed by Astro. To verify the release build and relationship rules:

```bash
uv sync --locked --project examples/foundations
npm run release:build
```

## Content model

- `src/data/book-map.ts` records the book's three parts, 12 chapters, and a curated topic outline.
- `src/data/topic-coverage.json` maps that outline's 95 entries to published lessons. Closely related entries can share a page; these entries are not the book's complete contents.
- `src/data/book-sections.json` records the audited source sections, embedded concepts, exact source locations, remaining work, and separate coverage dimensions. Containers are excluded from leaf-objective counts.
- `src/data/lesson-audit.json`, `proposed-concepts.json`, and `book-notebooks.json` record per-lesson findings, planned destinations/aliases, and pinned official notebook references. See the [audit guide](./docs/audits/README.md).
- `src/content/docs/concepts/*.mdx` contains concept pages. Its frontmatter supplies the concept ID, chapter, parent, prerequisites, and related links.
- `src/content/docs/paths/` provides three guided routes through the concepts; `src/content/docs/use-cases/` applies them to classification, research collection discovery, and RAG.
- `src/components/ConceptDirectory.astro` provides a chapter-grouped, filterable index. `src/components/ConceptPrimer.astro` offers optional completion tracking stored only in the reader's browser.
- `src/components/simulations/` contains browser-only embedding, attention, RAG, classification, clustering, prompt, multimodal, contrastive, adapter, and preference labs; the tokenization lab lives in `src/components/TokenizationLab.astro`. Experiments distinguish calculated teaching quantities from recorded model results and need no account or API key.
- `src/lib/graph.js` combines the book map and concept pages into the atlas data. It rejects duplicate IDs, missing targets, and relationship cycles during the production build.
- `src/pages/atlas-data.json.ts` exposes the validated graph as static JSON for the interactive view.
- `src/components/AtlasExplorer.astro` and `src/scripts/atlas-explorer.ts` provide the graph, filters, search-to-focus, topic inspector, shareable `?focus=` links, and a keyboard-accessible topic directory. The graph opens on a chapter overview on wide screens and a simpler part overview on narrow screens.
- `tests/content.test.mjs` checks the existing lesson structure and internal source links; it does not certify teaching depth.
- `npm run check:coverage` checks source mappings, anchors, page ranges, review evidence, stale source hashes, and report drift. `npm run coverage:report` regenerates the audit reports after reviewed changes.
- `scripts/check-code-examples.mjs` runs the 163 standard-library foundation examples, including all 76 existing exact-output comparisons. Historical model sketches retain their status; the three E2 reference lessons have recorded CPU tokenizer/training workflows.
- `scripts/check-internal-links.mjs` checks links and assets in the production build. The GitHub Pages workflow runs the complete `release:build` gate before deployment.

The book and its artwork are copyrighted. This project uses original explanations and visuals and links to the [authors' official notebooks](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models). It is not affiliated with the authors or publisher. See the [site sources page](./src/content/docs/about-and-sources.mdx) and [contribution guide](./CONTRIBUTING.md).

## GitHub Pages

The site is published at `https://shadril238.github.io/llm-concept-atlas/`. The [deployment workflow](./.github/workflows/deploy.yml) builds on pushes to `main`, runs the release checks, and publishes the static site. The repository’s **Settings → Pages → Build and deployment → Source** is set to **GitHub Actions**.

## License

The original project code, lessons, and visuals are available under the [MIT License](./LICENSE). The book *Hands-On Large Language Models*, its text, and its artwork are not part of this license. The attributed UCI sentiment subset is CC BY 4.0; see its [provenance and changes](./examples/data/manifests/sentiment.json). This is an independent companion project.

## E6 retrieval and multimodal learning

The atlas now has 112 concept lessons. Chapters 8–9 connect 23 detailed lessons to actual BM25/dense/reranked retrieval, six grounded-answer strategies, OpenCLIP preprocessing/patches, genuine BLIP-2, and visual chat. Explore [L12–L14](https://shadril238.github.io/llm-concept-atlas/labs/retrieval-and-multimodal/) and the [execution guide](examples/applications/retrieval/README.md). Four smaller CPU profiles and all five fresh-kernel notebooks have execution evidence; genuine BLIP-2 uses a separate heavy Linux CPU profile. Optional hosted/judge workflows are documented offline importers only. Independent teaching review remains pending.
