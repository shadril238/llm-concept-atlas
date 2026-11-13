# LLM Concept Atlas

**See it. Try it. Understand it.** An independent visual companion to *Hands-On Large Language Models* by Jay Alammar and Maarten Grootendorst.

**Phases 1–5 are implemented** in [the implementation plan](./plan.md). The site covers all 12 chapter outlines with 96 full concept lessons, an interactive relationship graph, 11 teaching labs, three guided learning paths, and classification and RAG use cases. Each lesson has a definition, theory, worked example, original diagram, runnable teaching code, use cases, limits, and self-checks. Final reader and editorial review is pending.

**Read the site:** [llm-concept-atlas on GitHub Pages](https://shadril238.github.io/llm-concept-atlas/).

## Run locally

Requires a recent Node.js version.

```bash
npm ci
npm run dev
```

The site uses the `/llm-concept-atlas/` base path. Open the local URL printed by Astro. To verify the release build and relationship rules:

```bash
npm run release:build
```

## Content model

- `src/data/book-map.ts` records the book's three parts, 12 chapters, and chapter topic outlines.
- `src/data/topic-coverage.json` maps all 95 chapter-topic entries to published concept lessons. Closely related entries can share a lesson.
- `src/content/docs/concepts/*.mdx` contains concept pages. Its frontmatter supplies the concept ID, chapter, parent, prerequisites, and related links.
- `src/content/docs/paths/` provides three guided routes through the concepts; `src/content/docs/use-cases/` applies them to classification and RAG.
- `src/components/ConceptDirectory.astro` provides a chapter-grouped, filterable index. `src/components/ConceptPrimer.astro` offers optional completion tracking stored only in the reader's browser.
- `src/components/simulations/` contains browser-only embedding, attention, RAG, classification, clustering, prompt, multimodal, contrastive, adapter, and preference labs; the tokenization lab lives in `src/components/TokenizationLab.astro`. All simulations use deterministic teaching data and need no account or API key.
- `src/lib/graph.js` combines the book map and concept pages into the atlas data. It rejects duplicate IDs, missing targets, and relationship cycles during the production build.
- `src/pages/atlas-data.json.ts` exposes the validated graph as static JSON for the interactive view.
- `src/components/AtlasExplorer.astro` and `src/scripts/atlas-explorer.ts` provide the graph, filters, search-to-focus, topic inspector, shareable `?focus=` links, and a keyboard-accessible topic directory. The graph opens on a chapter overview on wide screens and a simpler part overview on narrow screens.
- `tests/content.test.mjs` checks that every published concept remains a full lesson and that internal lesson links resolve.
- `scripts/check-code-examples.mjs` runs the 76 new Python teaching examples and compares their output with the lesson text.
- `scripts/check-internal-links.mjs` checks links and assets in the production build. The GitHub Pages workflow runs the complete `release:build` gate before deployment.

The book and its artwork are copyrighted. This project uses original explanations and visuals and links to the [authors' official notebooks](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models). It is not affiliated with the authors or publisher. See the [site sources page](./src/content/docs/about-and-sources.mdx) and [contribution guide](./CONTRIBUTING.md).

## GitHub Pages

The site is published at `https://shadril238.github.io/llm-concept-atlas/`. The [deployment workflow](./.github/workflows/deploy.yml) builds on pushes to `main`, runs the release checks, and publishes the static site. The repository’s **Settings → Pages → Build and deployment → Source** is set to **GitHub Actions**.

## License

The original project code, lessons, and visuals are available under the [MIT License](./LICENSE). The book *Hands-On Large Language Models*, its text, and its artwork are not part of this license. This is an independent companion project.
