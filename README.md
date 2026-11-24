# LLM Concept Atlas

**See it. Try it. Understand it.** An independent visual companion to *Hands-On Large Language Models* by Jay Alammar and Maarten Grootendorst.

**Current implementation:** E9 connects 113 concept lessons through shared search, eight study routes, prerequisite explanations and honest coverage displays. E10 review preparation is underway; the [review packet](./docs/reviews/e10-review-packet.md) and [release candidate](./docs/releases/release-candidate.md) describe the evidence and remaining gates. Start with the [learning hub](https://shadril238.github.io/llm-concept-atlas/learning-hub/) or [find a term](https://shadril238.github.io/llm-concept-atlas/search/). Independent teaching acceptance and supported GPU QLoRA verification remain open.

The original Phases 1–5 produced a **curated companion across all 12 chapters**: 96 concept lessons, an interactive relationship graph, 11 teaching simulations, three guided paths, and two use cases. The explanations and examples vary in depth; this is not yet exhaustive book coverage or a complete set of evaluated model workflows.

**Enhancements E0 and E1 are complete:** the [book gap report](./docs/audits/book-gap-report.md) inventories 231 source entries and distinguishes partial explanations, missing topics, and practical implementation gaps. The [lesson depth audit](./docs/audits/lesson-depth-report.md) records findings for every existing concept. The [example guide](./examples/README.md) provides locked environments, a 239-example registry, synchronized lesson/notebook source, and recorded CPU verification. Follow [enhancement.md](./enhancement.md) for the remaining enhancement phases; independent teaching acceptance and verification of the remaining model workflows are pending.

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

- `src/data/book-map.ts` records the three parts and twelve chapters; chapter objective displays come from the audited source inventory.
- The former topic map is archived at `docs/audits/legacy-topic-coverage.json` for historical reference and is no longer a runtime coverage source.
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

## E7 encoder and representation training

The atlas now contains **113 concept lessons**. Fifteen Chapter 10–11 lessons connect detailed theory to six real CPU training workflows: three embedding losses, trained-teacher Augmented SBERT, TSDAE/adaptation, repeated few-shot BERT/SetFit, MLM continued pretraining and complete NER through the reused E2 trainer. Explore [L15–L20](https://shadril238.github.io/llm-concept-atlas/labs/embedding-and-representation-training/), the [training path](https://shadril238.github.io/llm-concept-atlas/paths/from-pretrained-encoder-to-domain-model/) and the [execution guide](examples/training/representation/README.md). Actual failures, gradients, frozen hashes, split provenance, task scores and artifact reload are inspectable. GPU/book-scale benchmark configurations and independent teaching acceptance remain pending.


### E8 · generation adaptation and evaluation

[Explore L21–L23](https://shadril238.github.io/llm-concept-atlas/labs/generation-adaptation/): nine detailed Chapter12 lessons, instruction-mask/rank/memory/preference/evaluation interactives, actual full/LoRA SFT, scalar reward training and separate DPO, three fresh-kernel notebooks, genuine Linux CPU NF4 adapter training, fixed metrics and an offline human/judge importer. Follow the [generation path](https://shadril238.github.io/llm-concept-atlas/paths/from-instructions-to-preferences/) and [project](https://shadril238.github.io/llm-concept-atlas/use-cases/instruction-preference-adaptation/). Setup/provenance/limits: `examples/training/generation/README.md`; author evidence: `docs/reviews/e8-generation-adaptation.md`.

The tiny experiment visibly fails all eight generated exact-match tasks despite some improved likelihood/preference metrics. It is an instructional training study, not a production assistant or a general benchmark. Human/editorial review and supported GPU QLoRA/paged optimizer verification remain open; the full E8 completion gate is not marked passed. The attributed Dolly subset and derived preference data retain CC BY-SA3.0 separately from repository MIT code. Hosted judging remains documented/offline-import only.


### E9 · a connected learning experience

Use [search](https://shadril238.github.io/llm-concept-atlas/search/) for aliases, full names, curated tasks and Pagefind full-text results; filter by chapter, content type and concept difficulty. The glossary and graph use the same lexical ranking. [Eight learning routes](https://shadril238.github.io/llm-concept-atlas/learning-hub/) connect all 113 concepts to prerequisite sequences, labs and projects. Reading and practice checkmarks are browser-local, self-reported and migrated from the original progress key with its backup retained.

[Coverage](https://shadril238.github.io/llm-concept-atlas/coverage/) distinguishes mapped leaf objectives, independently accepted teaching and recorded practical execution. The audit remains the source of truth; E8 GPU QLoRA and independent review remain open. E9 adds 20 query regressions, relationship explanations and payload budgets; see `docs/reviews/e9-learning-experience.md` and `docs/performance/`. E10 independent teaching/learner acceptance is still pending. Its [reproducible evidence inventory](./docs/reviews/e10-evidence-audit.json) records 49/51 practical objectives with execution evidence after reconciling eight stale foundation statuses, while independent acceptance remains 0/160. Run `npm run evidence:report` after evidence changes; the release gate checks freshness.
