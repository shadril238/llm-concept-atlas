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
            { label: "Run verified examples", slug: "labs/first-verified-examples" },
            { label: "Foundation labs · L01–L06", slug: "labs/foundations" },
            { label: "Classification + topics · L07–L08", slug: "labs/classification-and-topics" },
            { label: "Prompting + workflows · L09–L11", slug: "labs/prompting-and-workflows" },
            { label: "Retrieval + multimodal · L12–L14", slug: "labs/retrieval-and-multimodal" },
            { label: "From pixels to multimodal answers", slug: "paths/from-pixels-to-multimodal-answers" },
            { label: "Generation training · L21–L23", slug: "labs/generation-adaptation" },
            { label: "Instructions to preferences", slug: "paths/from-instructions-to-preferences" },
            { label: "Encoder training · L15–L20", slug: "labs/embedding-and-representation-training" },
            { label: "Pretrained encoder to domain model", slug: "paths/from-pretrained-encoder-to-domain-model" },
            { label: "Concept glossary", slug: "glossary" },
            { label: "About and sources", slug: "about-and-sources" },
          ],
        },
        {
          label: "Use cases",
          items: [
            { label: "Instruction adaptation study", slug: "use-cases/instruction-preference-adaptation" },
            { label: "Text classification", slug: "use-cases/text-classification" },
            { label: "Research collection explorer", slug: "use-cases/research-collection-explorer" },
            { label: "Grounded document assistant", slug: "use-cases/rag-question-answering" },
            { label: "Image/text collection search", slug: "use-cases/image-text-collection-search" },
            { label: "Adapt a service-search encoder", slug: "use-cases/service-search-adaptation" },
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
