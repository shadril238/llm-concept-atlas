# llm-concept-atlas — implementation plan

**Status (2026-11-11): Phases 1 and 2 implemented locally.** The repository contains the Starlight site, chapter map, one full tokenization lesson with an interactive teaching demo, 19 introductory concept notes, an interactive atlas graph, and graph validation. The remaining introductory notes are content scaffolding; they are not the finished lessons. Phases 3–5 remain on the roadmap.

## 1. Goal

Build a public, interactive learning companion to *Hands-On Large Language Models* by Jay Alammar and Maarten Grootendorst. A reader should be able to search for a concept, understand it in plain language, manipulate a visual example, inspect a small code example, and see how it connects to other concepts.

The central product promise is: **see it, try it, understand it**. The site is an independent educational project, with original explanations and visuals and clear links to the book and its official notebooks.

## 2. Learning and design principles

1. **One concept per page.** Each page answers “what is it?”, “why does it matter?”, and “where is it used?” before introducing implementation detail.
2. **Show the process.** Readers can change inputs, parameters, or steps and see what changes. Interactive diagrams should explain a mechanism, not merely animate it.
3. **Use progressively deeper layers.** Start with a 30-second explanation, then a visual example, then a more precise explanation, then code.
4. **Connect ideas.** Every concept has a parent topic, related concepts, and prerequisite links. These connections power navigation and the atlas graph.
5. **Be honest about simplifications.** Label toy simulations and distinguish their behavior from a real model. Link to a runnable notebook when actual model computation is needed.
6. **Keep reading accessible.** Good contrast, readable type, keyboard controls, reduced-motion support, captions, and a text alternative for every graph or simulation.

## 3. Site structure

| Area | Purpose |
| --- | --- |
| Home | Clear starting points: “Start learning,” “Explore the atlas,” and “Find a concept.” Show a small interactive preview. |
| Learning paths | Guided sequences such as “How an LLM processes a sentence” and “How RAG answers a question.” |
| Chapter overviews | A short page for each chapter showing its purpose, key concepts, and place in the learning path. |
| Concepts | Individual term pages with explanation, visual, simulation, code, use cases, and links. |
| Atlas graph | Interactive map of all published concepts, their hierarchy, prerequisites, and related links. |
| Use cases | Problem-led pages such as classification, topic discovery, semantic search, RAG, and image captioning. |
| Labs | Small exercises and links to runnable Python notebooks. |
| Search | Full-text search across concepts, paths, use cases, and labs. |

The content follows the book's three parts: foundations (chapters 1–3), using pretrained models (chapters 4–9), and training or fine-tuning (chapters 10–12). Before writing pages, create a concept inventory from all 12 chapter outlines. Give each distinct concept a stable ID and place it in the hierarchy. The inventory is the coverage checklist; a public concept node must have at least a clear definition and a useful destination page.

## 4. Full concept lesson specification

Every published concept should work as a **self-contained lesson**, so someone can understand it without owning or opening the book. The book remains a deeper source and receives attribution. Page length can vary with topic complexity, but a definition and two short paragraphs do not count as a finished lesson.

Each full lesson follows this teaching sequence:

1. **Definition and intuition:** a precise definition, a plain-language explanation, and a concrete analogy or motivating question.
2. **Prerequisites:** a short list of concepts to know first, with links.
3. **Theory:** the mechanism step by step, with notation or an equation only when it improves understanding; define every symbol.
4. **Worked example:** trace one small input through the mechanism and explain each intermediate result.
5. **Visual and simulation:** an original labeled diagram and, where interaction adds value, controls that let the reader change input or parameters and observe the result.
6. **Implementation:** runnable code with dependencies, setup, input, expected behavior, and an explanation of the important lines. Link a notebook for GPU-heavy work.
7. **Use cases and decisions:** concrete problems it solves, when to use it, and when another approach is better.
8. **Limits and misconceptions:** where the simple mental model breaks down, typical errors, and what the concept does not imply.
9. **Check understanding:** two or three questions or a small exercise with answers or hints.
10. **Connections and sources:** parent, children, prerequisites, related terms, the relevant book chapter, and links to primary technical references.

Use tokenization as the first full-lesson reference. Mark pages as `primer` or `full` in frontmatter, and show that status to readers. The 19 remaining introductory notes are a starting inventory; expand them to this standard during Phase 3. A public release should not present primer pages as complete lessons.

## 5. Interactive atlas graph

### Data model

Extend Starlight's content frontmatter schema so concept pages contain structured metadata. Example:

```yaml
title: Attention
conceptId: attention
chapter: 3
part: foundations
summary: A mechanism that lets a token use information from other tokens.
parent: transformer
prerequisites: [tokens, embeddings]
related: [self-attention, query-key-value, kv-cache]
level: beginner
tags: [architecture, generation]
```

Generate the graph from this content at build time. The same IDs and links feed the graph, concept-page connection cards, breadcrumbs, and “learn next” suggestions. There are three visible edge types:

- **Contains:** parent → child. Every concept has exactly one parent; the three book parts and their chapter overview pages form the top of the hierarchy.
- **Learn before:** prerequisite → concept. These edges form a directed learning path.
- **Related:** a meaningful cross-link between concepts in different branches.

Validation must catch duplicate IDs, missing targets, invalid URLs, parent cycles, prerequisite cycles, and visible nodes without destination pages. Every published concept appears in the graph automatically; adding a page cannot silently leave it out.

### Graph experience

- Open on a **chapter-grouped overview** so the first screen stays legible. All concepts are present in the data; readers can expand chapters, zoom, pan, or switch to a focused neighborhood.
- Selecting a node highlights its parent, children, prerequisites, and related concepts. A side panel shows its plain-language definition, chapter, and “Open lesson” action.
- Provide filters for part, chapter, difficulty, and edge type; a graph search focuses a matching node. A selected concept has a shareable URL.
- Use clear labels and colors for edge types, with a legend that can be toggled. Cross-links should appear on selection or when enabled, so the full map does not become a tangle.
- On small screens, use a simplified focused graph plus a navigable relationship list. Supply the list on desktop too for keyboard and screen-reader access.
- Start with a hierarchical layout for parent/child edges. Use stable positions so the map does not jump unexpectedly when filters change.

Use **Cytoscape.js** for the graph viewport and selection behavior. Keep the source graph as plain typed data so the UI library can be changed without rewriting content.

## 6. First simulations

| Simulation | Reader controls | What the visual teaches |
| --- | --- | --- |
| Tokenization | Enter text and choose a tokenizer or example | A model sees token IDs, and different tokenizers split text differently. |
| Embedding similarity | Select words or short sentences | Similar meaning can place vectors near each other; 2D is an illustrative projection. |
| Attention | Select a token and step through a small sentence | Attention weights show which positions contribute to that token's next representation. |
| RAG pipeline | Ask a question and change the retrieved documents | Retrieval, ranking, context selection, and generation are separate steps. |

Build these as small React + SVG components embedded in MDX pages. Each should have a short “what to notice” prompt, reset control, explanatory labels, and a static text explanation. Keep the first simulations deterministic and runnable in the browser so visitors can learn without an account or API key. Add links to real-model Colab exercises where useful.

## 7. UI direction

Create a visual identity for **llm-concept-atlas** rather than copying the book's artwork:

- A calm, modern reference-site layout with strong typography, generous spacing, and restrained color. Use one accent color per book part so readers can recognize branches of the atlas.
- Home page: a concise hero, prominent search, three entry cards, a preview of the graph, and a suggested first path.
- Concept pages: a readable article column, a prominent interactive visual, sticky local navigation on wide screens, and connection cards near the end.
- Graph page: a full-width canvas with search and filters on one side and a concept inspector on the other. On mobile, controls become a drawer.
- Consistent visual language: token chips, vector arrows, document cards, and step markers mean the same thing across diagrams and simulations.
- Light and dark themes. Animate state changes briefly, while respecting the user's reduced-motion preference.

Create reusable components and design tokens before making many pages. Review the first home page, graph page, and concept page together so they feel like one product.

## 8. Technical architecture

| Concern | Choice | Reason |
| --- | --- | --- |
| Site and documentation | Astro + Starlight + MDX | Static content, navigation, code blocks, and a maintainable writing workflow. |
| Interactive lessons | React islands with TypeScript and SVG | Interactivity only where needed; reusable, testable simulation components. |
| Topic graph | Cytoscape.js | Pan, zoom, selection, and graph layout support. |
| Site search | Starlight's built-in Pagefind search | Indexes the built static pages without a search server. |
| Hosting | GitHub Pages via GitHub Actions | Source and public site can live in the same GitHub repository. |
| Heavy code | Python notebooks / Colab links | Keep GPU work outside the static website. |

Search for lessons and the book's **semantic search** lesson serve different purposes. The site search finds atlas pages; the semantic search demo teaches how retrieval works.

Suggested repository layout:

```text
llm-concept-atlas/
├── src/
│   ├── content/docs/
│   │   ├── chapters/
│   │   ├── concepts/
│   │   ├── paths/
│   │   ├── use-cases/
│   │   └── labs/
│   ├── components/
│   │   ├── graph/
│   │   └── simulations/
│   ├── lib/graph/
│   ├── pages/atlas.astro
│   ├── styles/
│   └── content.config.ts
├── tests/
├── .github/workflows/
├── README.md
├── CONTRIBUTING.md
└── plan.md
```

## 9. Implementation phases and completion gates

### Phase 1 — foundation and content map

- Scaffold Astro/Starlight; configure title, navigation, themes, repository links, and GitHub Pages base path.
- Inventory the concepts and chapter relationships across all 12 chapters. Create overview pages for the three parts and 12 chapters. Define stable IDs, a page template, and the extended frontmatter schema.
- Implement graph-data generation and validation before drawing the graph.
- **Done when:** the site builds; the concept inventory is reviewable; invalid relationships fail the build.

### Phase 2 — visual system and graph

- Design home, concept, and graph layouts with desktop and mobile states.
- Build graph overview, focus mode, filters, search-to-focus, inspector, shareable selected-node URL, and relationship list.
- **Done when:** every published concept appears in the graph, every graph node opens a page, and the graph works with mouse, touch, and keyboard-assisted list navigation.

### Phase 3 — first complete learning path

- Expand the 19 remaining introductory concept notes into full lessons from chapters 1–3 and 8, following every item in Section 4. Review the tokenization lesson against the same standard.
- Extend the tokenization teaching demo and build the embedding, attention, and RAG simulations, plus one guided path: “How an LLM processes a sentence.”
- Add the first use case pages for classification and RAG, with code links.
- **Done when:** a beginner can complete the path, use a simulation, search a term, and follow connections without encountering a one-line or unfinished lesson. All 20 pages are marked `full` after editorial review.

### Phase 4 — public release

- Add a second path: “From semantic search to RAG.”
- Verify mobile layout, contrast, keyboard access, reduced motion, captions, broken links, search results, and graph performance.
- Add README, contribution guide, source attribution, and a clear independent-project notice.
- Configure GitHub Actions deployment and publish GitHub Pages.
- **Done when:** the public site loads from the repository URL and its main learning flows work in a production build.

### Phase 5 — full book coverage

- Expand the concept catalog and graph across chapters 4–12. Each new public concept receives a full lesson, not only a glossary definition.
- Add labs for clustering, prompting, multimodal models, contrastive learning, LoRA/QLoRA, and preference tuning.
- Add short comprehension checks and optional local progress tracking after the core lessons are strong.
- **Done when:** every concept in the reviewed inventory has a self-contained full lesson and a valid place in the atlas graph.

## 10. Quality and maintenance

- **Content checks:** verify each page's summary, example, visual labels, code output, misconception, book chapter, and connections. Review explanations with beginners as well as technically experienced readers.
- **Automated checks:** TypeScript, production build, internal links, graph ID and cycle validation, and a few browser checks for search, graph selection, and a simulation.
- **Accessibility checks:** keyboard traversal, focus visibility, screen-reader relationship list, contrast, reduced motion, and mobile touch targets.
- **Performance checks:** lazy-load the graph and simulations; check the initial page load and test the graph with the full concept inventory.
- **Maintenance:** give pages a `lastReviewed` date for facts that can change. Keep fundamental explanations separate from current model or library examples.
- **Attribution:** cite the book and link to the official code repository. Write original text and graphics; do not commit the PDF, copied book figures, or large passages from the book.

## 11. First implementation task

Create the Starlight repository, the extended concept schema, the full chapter-level inventory, detailed metadata for the first 20 concepts, and a noninteractive prototype of the home, concept, and atlas pages. Review that structure before implementing the graph and simulations.

## Reference links

- Book code and notebooks: https://github.com/HandsOnLLM/Hands-On-Large-Language-Models
- Starlight setup: https://starlight.astro.build/getting-started/
- Starlight custom frontmatter: https://starlight.astro.build/reference/frontmatter/
- Starlight search: https://starlight.astro.build/guides/site-search/
- Cytoscape.js: https://js.cytoscape.org/
- Astro GitHub Pages deployment: https://docs.astro.build/en/guides/deploy/github/
