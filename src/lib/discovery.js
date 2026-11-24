/** Shared lexical ranking. Curated task phrases are explicit, not semantic search. */
export const normalize = (text = "") =>
  String(text)
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim()
    .replace(/\s+/g, " ");
const stop = new Set([
  "a",
  "an",
  "the",
  "how",
  "to",
  "with",
  "for",
  "my",
  "in",
  "of",
  "and",
  "i",
  "can",
]);
export function rankRecords(records, query = "", filters = {}) {
  const q = normalize(query),
    tokens = q.split(" ").filter((t) => t && !stop.has(t));
  return records
    .flatMap((r) => {
      if (filters.kind && filters.kind !== "all" && r.kind !== filters.kind)
        return [];
      if (
        filters.chapter &&
        filters.chapter !== "all" &&
        !(r.chapters ?? [r.chapter]).includes(Number(filters.chapter))
      )
        return [];
      if (filters.level && filters.level !== "all" && r.level !== filters.level)
        return [];
      const title = normalize(r.title),
        aliases = (r.aliases ?? []).map(normalize),
        tasks = (r.tasks ?? []).map(normalize);
      const hay = normalize(
        [
          r.title,
          r.summary,
          ...(r.aliases ?? []),
          ...(r.tasks ?? []),
          ...(r.tags ?? []),
        ].join(" "),
      );
      let score = 0,
        match = "Browse";
      if (q) {
        if (title === q) {
          score = 120;
          match = "Title";
        } else if (aliases.includes(q)) {
          score = 115;
          match = "Alias";
        } else if (tasks.includes(q)) {
          score = 105;
          match = "Task";
        } else if (
          q.length > 3 ? title.includes(q) : title.split(" ").includes(q)
        ) {
          score = 85;
          match = "Title";
        } else if (
          aliases.some((a) =>
            q.length > 3 ? a.includes(q) : a.split(" ").includes(q),
          )
        ) {
          score = 75;
          match = "Alias";
        } else if (
          tokens.length &&
          tokens.every((t) => hay.split(" ").includes(t))
        ) {
          score =
            50 + tokens.filter((t) => title.split(" ").includes(t)).length * 3;
          match = tasks.some((t) =>
            tokens.every((w) => t.split(" ").includes(w)),
          )
            ? "Task"
            : "Keywords";
        } else return [];
      }
      return [{ ...r, score, match }];
    })
    .sort((a, b) => b.score - a.score || a.title.localeCompare(b.title, "en"));
}
export function suggestions(records, query) {
  const q = normalize(query);
  if (q.length < 3) return [];
  const distance = (a, b) => {
    let row = Array.from({ length: b.length + 1 }, (_, i) => i);
    for (let i = 1; i <= a.length; i++) {
      const next = [i];
      for (let j = 1; j <= b.length; j++)
        next[j] = Math.min(
          next[j - 1] + 1,
          row[j] + 1,
          row[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1),
        );
      row = next;
    }
    return row[b.length];
  };
  return [...new Set(records.flatMap((r) => [r.title, ...(r.aliases ?? [])]))]
    .map((label) => ({ label, d: distance(q, normalize(label)) }))
    .filter((r) => r.d <= Math.min(2, Math.floor(q.length / 4)))
    .sort((a, b) => a.d - b.d || a.label.localeCompare(b.label))
    .slice(0, 3)
    .map((r) => r.label);
}
export function prerequisiteRoute(nodes, id) {
  const by = new Map(nodes.map((n) => [n.id, n])),
    visiting = new Set(),
    seen = new Set(),
    out = [];
  function visit(key) {
    if (visiting.has(key)) throw new Error("Prerequisite cycle");
    if (seen.has(key)) return;
    const n = by.get(key);
    if (!n) throw new Error("Unknown prerequisite " + key);
    visiting.add(key);
    for (const p of n.prerequisites ?? []) visit(p);
    visiting.delete(key);
    seen.add(key);
    out.push(n);
  }
  visit(id);
  return out;
}
/** Related links can be inbound: preserve the reason from the node that declared it. */
export function relationshipReason(nodes, from, to, type) {
  const a = nodes.find((n) => n.id === from),
    b = nodes.find((n) => n.id === to);
  if (!a || !b) return "";
  if (a.relationshipReasons?.[to]) return a.relationshipReasons[to];
  if (b.relationshipReasons?.[from]) return b.relationshipReasons[from];
  if (type === "contains")
    return `${b.title} is organized within ${a.title}. Book location and learning prerequisites are shown separately.`;
  return "";
}
