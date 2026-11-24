import { rankRecords, suggestions } from "../lib/discovery.js";
type RecordItem = {
  id: string;
  title: string;
  summary: string;
  href: string;
  kind: string;
  chapter?: number;
  chapters: number[];
  level: string;
  aliases: string[];
  tasks: string[];
  labHref?: string;
  pathHref?: string;
  projectHref?: string;
  match?: string;
};
let catalog: Promise<RecordItem[]> | undefined,
  pagefind: Promise<any> | undefined;
function records(base: string) {
  return (catalog ??= fetch(base + "discovery.json")
    .then((r) => {
      if (!r.ok) throw Error("catalog");
      return r.json();
    })
    .catch((e) => {
      catalog = undefined;
      throw e;
    }));
}
const text = (tag: string, value: string, className = "") => {
  const el = document.createElement(tag);
  el.textContent = value;
  el.className = className;
  return el;
};
for (const root of document.querySelectorAll<HTMLElement>("[data-discovery]")) {
  const base = root.dataset.base ?? "/",
    mode = root.dataset.mode;
  const input = root.querySelector<HTMLInputElement>("[data-query]")!,
    chapter = root.querySelector<HTMLSelectElement>("[data-chapter]")!,
    kind = root.querySelector<HTMLSelectElement>("[data-kind]"),
    level = root.querySelector<HTMLSelectElement>("[data-level]")!,
    status = root.querySelector<HTMLElement>("[data-status]")!,
    results = root.querySelector<HTMLElement>("[data-results]")!,
    more = root.querySelector<HTMLButtonElement>("[data-more]")!,
    suggested = root.querySelector<HTMLElement>("[data-suggestions]")!;
  const fullSection = root.querySelector<HTMLElement>(
      "[data-fulltext-section]",
    )!,
    full = root.querySelector<HTMLElement>("[data-fulltext]")!,
    fullStatus = root.querySelector<HTMLElement>("[data-fulltext-status]")!,
    retry = root.querySelector<HTMLButtonElement>("[data-retry]")!;
  let limit = 16,
    revision = 0,
    timer: ReturnType<typeof setTimeout>,
    lastMatches: RecordItem[] = [];
  const filters = () => ({
    chapter: chapter.value,
    kind: mode === "concept" ? "concept" : (kind?.value ?? "all"),
    level: level.value,
  });
  function card(r: RecordItem) {
    const article = text("article", "", "lookup-card"),
      a = text("a", r.title) as HTMLAnchorElement;
    a.href = base + r.href.replace(/^\/+/, "");
    const meta = [
      r.kind === "concept"
        ? "Concept"
        : r.kind === "lab"
          ? "Lab"
          : r.kind === "project"
            ? "Project"
            : r.kind === "path"
              ? "Learning path"
              : r.kind,
      r.chapter
        ? "Chapter " + r.chapter
        : r.chapters?.length
          ? "Chapters " + r.chapters.join(", ")
          : "",
      r.level,
      r.match && input.value ? r.match + " match" : "",
    ]
      .filter(Boolean)
      .join(" · ");
    const h = text("h3", "");
    h.append(a);
    article.append(text("small", meta), h, text("p", r.summary));
    if (r.aliases?.length)
      article.append(
        text("p", "Also known as: " + r.aliases.join(" · "), "lookup-aliases"),
      );
    if (r.kind === "concept") {
      const nav = text("nav", "", "lookup-links");
      nav.setAttribute("aria-label", r.title + " next steps");
      for (const [label, path] of [
        ["Topic map", "atlas/?focus=" + encodeURIComponent(r.id)],
        ["Connected lab", r.labHref],
        ["Learning path", r.pathHref],
      ])
        if (path) {
          const link = text("a", label!) as HTMLAnchorElement;
          link.href = base + path;
          nav.append(link);
        }
      article.append(nav);
    }
    return article;
  }
  function renderCards() {
    results.replaceChildren(...lastMatches.slice(0, limit).map(card));
    more.hidden = lastMatches.length <= limit;
  }
  async function update() {
    const generation = ++revision,
      q = input.value.trim();
    let all: RecordItem[];
    try {
      all = await records(base);
    } catch {
      if (generation !== revision) return;
      status.textContent =
        "Search could not load. Use the topic hierarchy or try clearing the filters to retry.";
      return;
    }
    if (generation !== revision) return;
    let matches = rankRecords(all, q, filters()) as RecordItem[];
    if (
      !q &&
      mode === "global" &&
      chapter.value === "all" &&
      level.value === "all" &&
      kind?.value === "all"
    )
      matches = matches.filter((r) => r.kind === "path");
    lastMatches = matches;
    renderCards();
    suggested.replaceChildren();
    status.textContent = matches.length
      ? `${matches.length} ${q ? (matches.length === 1 ? "match" : "matches") : matches.length === 1 ? "destination" : "destinations"}${matches.length > limit ? "; showing " + limit : ""}.`
      : mode === "concept"
        ? "No concepts match these filters. Clear a filter or try an acronym, full name or task."
        : "No concept or route matches in this lookup. Check Inside the lessons below, clear a filter, or try another term.";
    if (!matches.length && q)
      for (const label of suggestions(all, q)) {
        const button = text(
          "button",
          "Try “" + label + "”",
        ) as HTMLButtonElement;
        button.type = "button";
        button.addEventListener("click", () => {
          input.value = label;
          limit = 16;
          void update();
          input.focus();
        });
        suggested.append(button);
      }
    if (mode === "search") {
      const url = new URL(location.href);
      for (const [k, v] of Object.entries({
        q,
        chapter: chapter.value,
        type: kind?.value ?? "all",
        level: level.value,
      }))
        if (v && v !== "all") url.searchParams.set(k, v);
        else url.searchParams.delete(k);
      history.replaceState(null, "", url);
    }
    if (mode === "concept" || !q) {
      fullSection.hidden = true;
      return;
    }
    fullSection.hidden = false;
    full.replaceChildren();
    retry.hidden = true;
    fullStatus.textContent = "Searching lesson text…";
    try {
      const api = await (pagefind ??= import(
        /* @vite-ignore */ base + "pagefind/pagefind.js"
      ).catch((e) => {
        pagefind = undefined;
        throw e;
      }));
      const pfFilters: Record<string, string> = {};
      if (kind?.value && kind.value !== "all") pfFilters.kind = kind.value;
      if (chapter.value !== "all") pfFilters.chapter = chapter.value;
      if (level.value !== "all") pfFilters.level = level.value;
      const response = await api.search(q, { filters: pfFilters });
      const hits = await Promise.all(
        response.results.slice(0, 8).map((r: any) => r.data()),
      );
      if (generation !== revision) return;
      fullStatus.textContent = `${response.results.length} full-text results${response.results.length > 8 ? "; showing 8" : ""}.`;
      for (const hit of hits) {
        const url = new URL(hit.url, location.origin);
        if (url.origin !== location.origin || !url.pathname.startsWith(base))
          continue;
        const el = text("article", "", "lookup-card"),
          h = text("h4", ""),
          a = text("a", hit.meta.title ?? "Lesson") as HTMLAnchorElement;
        a.href = url.href;
        h.append(a);
        el.append(
          h,
          text(
            "p",
            String(hit.excerpt ?? "")
              .replace(/<[^>]*>/g, "")
              .replace(/&amp;/g, "&")
              .replace(/&quot;/g, '"'),
          ),
        );
        full.append(el);
      }
    } catch {
      if (generation !== revision) return;
      fullStatus.textContent =
        "Full-text search is temporarily unavailable. Concept and route matches remain available.";
      retry.hidden = false;
    }
  }
  function schedule() {
    clearTimeout(timer);
    revision++;
    limit = 16;
    timer = setTimeout(() => void update(), 250);
  }
  input.addEventListener("input", schedule);
  for (const control of [chapter, kind, level])
    control?.addEventListener("change", () => {
      limit = 16;
      void update();
    });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !root.closest("dialog")) {
      input.value = "";
      void update();
    }
  });
  root.querySelector("[data-reset]")?.addEventListener("click", () => {
    clearTimeout(timer);

    chapter.value = "all";
    if (kind) kind.value = "all";
    level.value = "all";
    limit = 16;
    void update();
    input.focus();
  });
  more.addEventListener("click", () => {
    limit += 16;
    renderCards();
    status.textContent = `Showing ${Math.min(limit, lastMatches.length)} of ${lastMatches.length} matches.`;
  });
  retry.addEventListener("click", () => void update());
  root.addEventListener("discovery-open", () => {
    void update();
    input.focus();
  });
  if (mode !== "global") {
    const params = new URLSearchParams(location.search);
    input.value = params.get("q") ?? "";
    for (const [control, key] of [
      [chapter, "chapter"],
      [kind, "type"],
      [level, "level"],
    ] as const)
      if (control) {
        const value = params.get(key);
        if (value && [...control.options].some((o) => o.value === value))
          control.value = value;
      }
    void update();
  }
}
