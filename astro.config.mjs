// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";
import react from "@astrojs/react";

// https://astro.build/config
export default defineConfig({
  site: process.env.SITE_URL ?? "https://shadril238.github.io",
  base: "/llm-concept-atlas",
  integrations: [
    react(),
    starlight({
      title: "LLM Concept Atlas",
      description:
        "A visual field guide to language models, from first token to fine-tuning.",
      customCss: ["./src/styles/custom.css"],
      sidebar: [
        {
          label: "Start here",
          items: [
            { label: "Overview", slug: "index" },
            { label: "Explore the atlas", slug: "atlas" },
            { label: "Concept directory", slug: "concept-directory" },
            { label: "How an LLM processes a sentence", slug: "paths/how-an-llm-processes-a-sentence" },
            { label: "From semantic search to RAG", slug: "paths/from-semantic-search-to-rag" },
            { label: "From applications to fine-tuning", slug: "paths/from-applications-to-fine-tuning" },
            { label: "About and sources", slug: "about-and-sources" },
          ],
        },
        {
          label: "Use cases",
          items: [
            { label: "Text classification", slug: "use-cases/text-classification" },
            { label: "RAG question answering", slug: "use-cases/rag-question-answering" },
          ],
        },
        {
          label: "Part I · Foundations",
          items: [
            { slug: "parts/foundations" },
            { autogenerate: { directory: "chapters/foundations" } },
          ],
        },
        {
          label: "Part II · Applications",
          items: [
            { slug: "parts/applications" },
            { autogenerate: { directory: "chapters/applications" } },
          ],
        },
        {
          label: "Part III · Training",
          items: [
            { slug: "parts/training" },
            { autogenerate: { directory: "chapters/training" } },
          ],
        },
      ],
    }),
  ],
});
