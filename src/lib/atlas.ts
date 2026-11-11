import { getCollection } from "astro:content";
import { parts, chapters } from "../data/book-map";
import { buildAtlasGraph } from "./graph.js";

export interface AtlasNode {
  id: string;
  kind: "part" | "chapter" | "concept";
  title: string;
  summary: string;
  parent: string | null;
  href: string;
  part: string;
  chapter?: number;
  prerequisites: string[];
  related: string[];
  level?: string;
  lessonStatus?: "primer" | "full";
  tags?: string[];
}

export interface AtlasGraph {
  nodes: AtlasNode[];
  edges: {
    source: string;
    target: string;
    type: "contains" | "prerequisite" | "related";
  }[];
}

export async function getAtlasGraph() {
  const entries = await getCollection("docs");
  return buildAtlasGraph(parts, chapters, entries) as AtlasGraph;
}
