export function rankingMetrics(ranking, relevance, k = 3) {
  if (!Number.isInteger(k) || k < 1 || new Set(ranking).size !== ranking.length)
    throw Error("Positive cutoff and unique document IDs required.");
  if (Object.values(relevance).some((v) => !Number.isInteger(v) || v < 0))
    throw Error("Nonnegative integer grades required.");
  const total = Object.values(relevance).filter((v) => v > 0).length;
  let hits = 0,
    dcg = 0,
    rr = 0;
  const terms = [];
  ranking.slice(0, k).forEach((id, i) => {
    const grade = relevance[id] ?? 0;
    if (grade > 0) {
      hits++;
      terms.push(hits / (i + 1));
      if (!rr) rr = 1 / (i + 1);
    }
    dcg += (2 ** grade - 1) / Math.log2(i + 2);
  });
  const ideal = Object.values(relevance)
    .sort((a, b) => b - a)
    .slice(0, k)
    .reduce((sum, g, i) => sum + (2 ** g - 1) / Math.log2(i + 2), 0);
  return {
    precision: hits / k,
    recall: total ? hits / total : null,
    ap: total ? terms.reduce((a, b) => a + b, 0) / total : null,
    rr: total ? rr : null,
    ndcg: ideal ? dcg / ideal : null,
    hits,
    relevant_documents: total,
    ap_terms: terms,
    dcg,
    idcg: ideal,
  };
}
export function makeChunks(text, size = 2, overlap = 1, mode = "sentence") {
  if (
    !Number.isInteger(size) ||
    !Number.isInteger(overlap) ||
    size < 1 ||
    overlap < 0 ||
    overlap >= size
  )
    throw Error("Require size>0 and 0≤overlap<size.");
  const units = [];
  let start = 0;
  if (mode === "word")
    for (const match of text.matchAll(/\S+/gu))
      units.push([match.index, match.index + match[0].length]);
  else if (mode === "sentence")
    for (const match of text.matchAll(/(?<=[.!?])\s+|$/gu)) {
      let end = match.index,
        left = start;
      while (left < end && /\s/u.test(text[left])) left++;
      while (end > left && /\s/u.test(text[end - 1])) end--;
      if (end > left) units.push([left, end]);
      start = match.index + match[0].length;
    }
  else throw Error("Unknown chunk mode.");
  const result = [];
  let position = 0;
  while (position < units.length) {
    const end = Math.min(position + size, units.length),
      a = units[position][0],
      b = units[end - 1][1];
    result.push({
      start: Array.from(text.slice(0, a)).length,
      end: Array.from(text.slice(0, b)).length,
      text: text.slice(a, b),
    });
    if (end === units.length) break;
    position = end - overlap;
  }
  return result;
}
export function candidateSoftmax(scores, scale = 1) {
  if (
    !scores.length ||
    !Number.isFinite(scale) ||
    !scores.every(Number.isFinite)
  )
    throw Error("Finite scores and nonempty candidates required.");
  const top = Math.max(...scores.map((v) => v * scale)),
    e = scores.map((v) => Math.exp(v * scale - top)),
    sum = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / sum);
}
export function contrastiveLoss(logits) {
  const n = logits.length;
  if (n < 2 || logits.some((r) => r.length !== n || !r.every(Number.isFinite)))
    throw Error("Finite square paired matrix required.");
  const direction = (rows) =>
    rows.map((row, i) => {
      const top = Math.max(...row);
      return (
        top + Math.log(row.reduce((s, v) => s + Math.exp(v - top), 0)) - row[i]
      );
    });
  const a = direction(logits),
    b = direction(logits.map((_, i) => logits.map((row) => row[i])));
  return {
    image_to_text: a.reduce((x, y) => x + y, 0) / n,
    text_to_image: b.reduce((x, y) => x + y, 0) / n,
    loss:
      (a.reduce((x, y) => x + y, 0) + b.reduce((x, y) => x + y, 0)) / (2 * n),
    image_terms: a,
    text_terms: b,
  };
}
export function citationMetrics(claims) {
  for (const c of claims) {
    if (
      typeof c.fully_supported_by_citations !== "boolean" ||
      (c.fully_supported_by_citations && !c.citations.length) ||
      c.citations.some((v) => typeof v.supports !== "boolean")
    )
      throw Error("Explicit human link and joint-support judgments required.");
  }
  const links = claims.flatMap((c) => c.citations);
  return {
    citation_precision: links.length
      ? links.filter((v) => v.supports).length / links.length
      : null,
    citation_recall: claims.length
      ? claims.filter((c) => c.fully_supported_by_citations).length /
        claims.length
      : null,
    claims: claims.length,
    citations: links.length,
  };
}
