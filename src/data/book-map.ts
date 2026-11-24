/** A curated navigation map, based on the 2024 first edition's chapter outline. */
export type PartId = "foundations" | "applications" | "training";

export interface Chapter {
  number: number;
  title: string;
  summary: string;
  part: PartId;
}

export interface Part {
  id: PartId;
  number: string;
  title: string;
  shortTitle: string;
  summary: string;
  chapters: number[];
}

export const parts: Part[] = [
  {
    id: "foundations",
    number: "I",
    title: "Understanding Language Models",
    shortTitle: "Foundations",
    summary:
      "Build a mental model of tokens, embeddings, attention, and the transformer.",
    chapters: [1, 2, 3],
  },
  {
    id: "applications",
    number: "II",
    title: "Using Pretrained Language Models",
    shortTitle: "Applications",
    summary:
      "Put models to work in classification, search, generation, and multimodal tasks.",
    chapters: [4, 5, 6, 7, 8, 9],
  },
  {
    id: "training",
    number: "III",
    title: "Training and Fine-Tuning Language Models",
    shortTitle: "Training",
    summary:
      "Adapt representations and generators with contrastive learning and fine-tuning.",
    chapters: [10, 11, 12],
  },
];

export const chapters: Chapter[] = [
  {
    number: 1,
    title: "An Introduction to Large Language Models",
    part: "foundations",
    summary:
      "The big picture: language AI, model families, and what language models can do.",
  },
  {
    number: 2,
    title: "Tokens and Embeddings",
    part: "foundations",
    summary:
      "How text becomes tokens and how models turn those tokens into useful vectors.",
  },
  {
    number: 3,
    title: "Looking Inside Large Language Models",
    part: "foundations",
    summary:
      "Follow information through a transformer, from input tokens to a sampled output.",
  },
  {
    number: 4,
    title: "Text Classification",
    part: "applications",
    summary:
      "Assign useful labels to text with representations or generative models.",
  },
  {
    number: 5,
    title: "Text Clustering and Topic Modeling",
    part: "applications",
    summary:
      "Discover patterns in a collection of documents without predefined labels.",
  },
  {
    number: 6,
    title: "Prompt Engineering",
    part: "applications",
    summary:
      "Shape model behavior with instructions, examples, and output constraints.",
  },
  {
    number: 7,
    title: "Advanced Text Generation Techniques and Tools",
    part: "applications",
    summary:
      "Combine model calls with templates, memory, chains, and tool-using agents.",
  },
  {
    number: 8,
    title: "Semantic Search and Retrieval-Augmented Generation",
    part: "applications",
    summary:
      "Find relevant information by meaning and ground generation in retrieved sources.",
  },
  {
    number: 9,
    title: "Multimodal Large Language Models",
    part: "applications",
    summary:
      "Connect language with images through shared representations and multimodal models.",
  },
  {
    number: 10,
    title: "Creating Text Embedding Models",
    part: "training",
    summary:
      "Train sentence representations and understand contrastive objectives.",
  },
  {
    number: 11,
    title: "Fine-Tuning Representation Models for Classification",
    part: "training",
    summary:
      "Adapt an encoder to classification and named-entity recognition tasks.",
  },
  {
    number: 12,
    title: "Fine-Tuning Generation Models",
    part: "training",
    summary:
      "Understand supervised fine-tuning, efficient adapters, evaluation, and preferences.",
  },
];

export const chapterId = (number: number) =>
  `chapter-${String(number).padStart(2, "0")}`;
export const chapterSlug = (chapter: Chapter) =>
  `chapters/${chapter.part}/${String(chapter.number).padStart(2, "0")}`;
export const chapterByNumber = (number: number) =>
  chapters.find((chapter) => chapter.number === number);
