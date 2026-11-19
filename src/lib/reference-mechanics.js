export function loraTrace(alpha, rank, input = [2, 1]) {
  const compressed = 0.3 * input[0] - 0.1 * input[1];
  const delta = [0.1, 0.2].map((value) => (alpha / rank) * value * compressed);
  return {
    compressed,
    delta,
    output: input.map((value, i) => value + delta[i]),
  };
}
export function bioSpans(tags) {
  const result = [];
  let active = null;
  let start = 0;
  [...tags, "O"].forEach((tag, index) => {
    const [prefix, kind] = tag === "O" ? ["O", null] : tag.split("-");
    const continues = prefix === "I" && kind === active;
    if (active !== null && !continues) {
      result.push([active, start, index]);
      active = null;
    }
    if (prefix !== "O" && !continues) {
      active = kind;
      start = index;
    }
  });
  return result;
}
export function spanScore(gold, prediction) {
  const a = bioSpans(gold).map((span) => JSON.stringify(span));
  const b = bioSpans(prediction).map((span) => JSON.stringify(span));
  const tp = b.filter((span) => a.includes(span)).length;
  const precision = b.length ? tp / b.length : 0;
  const recall = a.length ? tp / a.length : 0;
  return {
    tp,
    precision,
    recall,
    f1:
      precision + recall ? (2 * precision * recall) / (precision + recall) : 0,
  };
}
