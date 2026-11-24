import { getCollection } from "astro:content";
import routes from "../data/learning-routes.json";
import terms from "../data/discovery-terms.json";
export const learningRoutes = routes;
const extra: Record<string, { aliases: string[]; tasks: string[] }> = terms;
export function routeForChapter(chapter?: number) {
  return routes.find((r) => r.chapters.includes(chapter ?? 0));
}
export function enrichConcept(data: any) {
  const route = routeForChapter(data.chapter),
    term = extra[data.conceptId];
  return {
    ...data,
    aliases: [...new Set([...(data.aliases ?? []), ...(term?.aliases ?? [])])],
    tasks: term?.tasks ?? [],
    labHref: (data.labHref ?? route?.labHref)?.replace(/^\/+/, ""),
    pathHref: route?.pathHref,
    projectHref: route?.projectHref,
  };
}
export async function discoveryRecords() {
  const entries = await getCollection("docs");
  return entries
    .filter(
      (e) =>
        ![
          "search",
          "concept-directory",
          "glossary",
          "learning-hub",
          "coverage",
        ].includes(e.id),
    )
    .map((entry) => {
      const d = enrichConcept({
        ...entry.data,
        chapter:
          entry.data.chapter ??
          (entry.id.startsWith("chapters/")
            ? Number(entry.id.split("/").at(-1))
            : undefined),
      });
      const kind = entry.id.startsWith("labs/")
        ? "lab"
        : d.kind === "use-case"
          ? "project"
          : ["part", "atlas"].includes(d.kind)
            ? "guide"
            : (d.kind ?? "guide");
      const associated = routes.filter((r) =>
        [
          r.labHref.split("#")[0],
          r.pathHref.split("#")[0],
          r.projectHref.split("#")[0],
        ].includes(entry.id + "/"),
      );
      return {
        id: d.conceptId ?? entry.id,
        title: d.title,
        summary: d.summary ?? d.description ?? "",
        href: entry.id === "index" ? "" : entry.id + "/",
        kind,
        chapter: d.chapter,
        chapters: d.chapter
          ? [d.chapter]
          : [...new Set(associated.flatMap((r) => r.chapters))],
        level: d.level ?? "",
        aliases: d.aliases,
        tasks: d.tasks,
        tags: d.tags ?? [],
        labHref: d.labHref,
        pathHref: d.pathHref,
        projectHref: d.projectHref,
      };
    });
}
