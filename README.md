# LLM Concept Atlas

**See it. Try it. Understand it.** An independent visual companion to *Hands-On Large Language Models* by Jay Alammar and Maarten Grootendorst.

This repository has completed **Phases 1–3** of [the implementation plan](./plan.md): a Starlight foundation, an outline of all 12 chapters, an interactive concept atlas, 20 full lessons, four interactive teaching demos, a guided learning path, and classification and RAG use-case pages. The current full lessons cover chapters 1–3 and 8; the remaining chapter outlines are a roadmap for later content.

## Run locally

Requires a recent Node.js version.

```bash
npm ci
npm run dev
```

The site uses the `/llm-concept-atlas/` base path. Open the local URL printed by Astro. To verify the release build and relationship rules:

```bash
npm run test:graph
npm run test:content
npm run check
npm run build
```

## Content model

- `src/data/book-map.ts` records the book's three parts, 12 chapters, and chapter topic outlines.
- `src/content/docs/concepts/*.mdx` contains concept pages. Its frontmatter supplies the concept ID, chapter, parent, prerequisites, and related links.
- `src/content/docs/paths/` provides the first guided route through the concepts; `src/content/docs/use-cases/` applies them to classification and RAG.
- `src/components/simulations/` contains the browser-only embedding, attention, and RAG labs; the tokenization lab lives in `src/components/TokenizationLab.astro`. All simulations use deterministic teaching data and need no account or API key.
- `src/lib/graph.js` combines the book map and concept pages into the atlas data. It rejects duplicate IDs, missing targets, and relationship cycles during the production build.
- `src/pages/atlas-data.json.ts` exposes the validated graph as static JSON for the interactive view.
- `src/components/AtlasExplorer.astro` and `src/scripts/atlas-explorer.ts` provide the graph, filters, search-to-focus, topic inspector, shareable `?focus=` links, and a keyboard-accessible topic directory. The graph opens on a chapter overview on wide screens and a simpler part overview on narrow screens.
- `tests/content.test.mjs` checks that every published concept remains a full lesson and that internal lesson links resolve.

The book and its artwork are copyrighted. This project uses original explanations and visuals and links to the [authors' official notebooks](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models). It is not affiliated with the authors or publisher.

## GitHub Pages setup later

The repository slug is configured as `llm-concept-atlas`, and the default site URL is `https://shadril238.github.io`. GitHub Pages deployment is planned for Phase 4; pushing this repository does not publish the site yet.

## License

The original project code, lessons, and visuals are available under the [MIT License](./LICENSE). The book *Hands-On Large Language Models*, its text, and its artwork are not part of this license. This is an independent companion project.
