import { getCollection } from "astro:content";
import { parts, chapters } from "../data/book-map";
import relationshipNotes from "../data/relationship-notes.json";
import { enrichConcept } from "./learning-catalog";
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
  aliases?: string[];
  relationshipReasons?: Record<string, string>;
  labHref?: string;
  pathHref?: string;
  projectHref?: string;
  tasks?: string[];
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
  const enriched = entries.map((e) => ({ ...e, data: enrichConcept(e.data) }));
  const by = new Map(
    enriched
      .filter((e) => e.data.conceptId)
      .map((e) => [e.data.conceptId, e.data]),
  );
  const notes: Record<string, string> = relationshipNotes;
  for (const entry of enriched) {
    const d = entry.data;
    if (!d.conceptId) continue;
    d.relationshipReasons = { ...(d.relationshipReasons ?? {}) };
    for (const id of [...(d.prerequisites ?? []), ...(d.related ?? [])]) {
      d.relationshipReasons[id] ??=
        by.get(id)?.relationshipReasons?.[d.conceptId] ??
        notes[[d.conceptId, id].sort().join("|")];
      if (!d.relationshipReasons[id])
        throw new Error(
          "Missing relationship explanation: " + d.conceptId + " / " + id,
        );
    }
  }
  return buildAtlasGraph(parts, chapters, enriched) as AtlasGraph;
}
