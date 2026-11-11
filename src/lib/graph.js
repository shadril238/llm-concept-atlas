/**
 * Build the atlas from Starlight content. Kept independent of Astro so the
 * relationship rules can be checked with Node's built-in test runner.
 */
export function buildAtlasGraph(parts, chapters, entries) {
  const errors = [];
  const nodes = [];
  const byId = new Map();
  const add = (node) => {
    if (byId.has(node.id)) errors.push(`Duplicate atlas ID: ${node.id}`);
    else {
      byId.set(node.id, node);
      nodes.push(node);
    }
  };

  for (const part of parts) {
    add({
      id: part.id,
      kind: "part",
      title: part.shortTitle,
      summary: part.summary,
      parent: null,
      href: `/parts/${part.id}/`,
      part: part.id,
      prerequisites: [],
      related: [],
    });
  }
  for (const chapter of chapters) {
    const id = `chapter-${String(chapter.number).padStart(2, "0")}`;
    if (!byId.has(chapter.part))
      errors.push(`${id} has unknown part ${chapter.part}`);
    add({
      id,
      kind: "chapter",
      title: chapter.title,
      summary: chapter.summary,
      parent: chapter.part,
      href: `/chapters/${chapter.part}/${String(chapter.number).padStart(2, "0")}/`,
      part: chapter.part,
      chapter: chapter.number,
      prerequisites: [],
      related: [],
    });
  }

  const docsById = new Map(entries.map((entry) => [entry.id, entry]));
  for (const part of parts) {
    if (!docsById.has(`parts/${part.id}`))
      errors.push(`Missing part overview page: ${part.id}`);
  }
  for (const chapter of chapters) {
    const slug = `chapters/${chapter.part}/${String(chapter.number).padStart(2, "0")}`;
    if (!docsById.has(slug))
      errors.push(`Missing chapter overview page: ${slug}`);
  }

  for (const entry of entries.filter((item) => item.data.kind === "concept")) {
    const data = entry.data;
    if (!data.conceptId) {
      errors.push(`${entry.id} has no conceptId`);
      continue;
    }
    if (entry.id !== `concepts/${data.conceptId}`)
      errors.push(
        `${data.conceptId} file path must be concepts/${data.conceptId}`,
      );
    const chapter = chapters.find((item) => item.number === data.chapter);
    if (!chapter)
      errors.push(`${data.conceptId} has unknown chapter ${data.chapter}`);
    else if (chapter.part !== data.part)
      errors.push(`${data.conceptId} has mismatched part ${data.part}`);
    if (!data.summary?.trim()) errors.push(`${data.conceptId} needs a summary`);
    if (!data.parent) errors.push(`${data.conceptId} needs a parent`);
    add({
      id: data.conceptId,
      kind: "concept",
      title: data.title,
      summary: data.summary,
      parent: data.parent,
      href: `/${entry.id}/`,
      part: data.part,
      chapter: data.chapter,
      prerequisites: data.prerequisites ?? [],
      related: data.related ?? [],
      level: data.level,
      lessonStatus: data.lessonStatus ?? "primer",
      tags: data.tags ?? [],
    });
  }

  for (const node of nodes) {
    if (node.parent && !byId.has(node.parent))
      errors.push(`${node.id} has unknown parent ${node.parent}`);
    for (const prerequisite of node.prerequisites) {
      if (!byId.has(prerequisite))
        errors.push(`${node.id} has unknown prerequisite ${prerequisite}`);
      if (prerequisite === node.id)
        errors.push(`${node.id} cannot require itself`);
    }
    for (const related of node.related) {
      if (!byId.has(related))
        errors.push(`${node.id} has unknown related concept ${related}`);
      if (related === node.id)
        errors.push(`${node.id} cannot relate to itself`);
    }
    if (!node.href) errors.push(`${node.id} has no destination`);
  }

  function checkCycles(edgeName, nextIds) {
    const visiting = new Set();
    const visited = new Set();
    const visit = (id, path) => {
      if (visiting.has(id)) {
        errors.push(`${edgeName} cycle: ${[...path, id].join(" → ")}`);
        return;
      }
      if (visited.has(id)) return;
      const node = byId.get(id);
      if (!node) return;
      visiting.add(id);
      for (const next of nextIds(node)) visit(next, [...path, id]);
      visiting.delete(id);
      visited.add(id);
    };
    for (const node of nodes) visit(node.id, []);
  }
  checkCycles("Parent", (node) => (node.parent ? [node.parent] : []));
  checkCycles("Prerequisite", (node) => node.prerequisites);

  if (errors.length)
    throw new Error(`Atlas validation failed:\n- ${errors.join("\n- ")}`);
  return {
    nodes,
    edges: nodes.flatMap((node) => [
      ...(node.parent
        ? [{ source: node.parent, target: node.id, type: "contains" }]
        : []),
      ...node.prerequisites.map((source) => ({
        source,
        target: node.id,
        type: "prerequisite",
      })),
      ...node.related.map((target) => ({
        source: node.id,
        target,
        type: "related",
      })),
    ]),
  };
}
