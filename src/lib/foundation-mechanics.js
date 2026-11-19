export function softmax(logits, temperature = 1) {
  if (
    !logits.length ||
    !Number.isFinite(temperature) ||
    temperature <= 0 ||
    logits.some((x) => !Number.isFinite(x))
  )
    throw new Error("Finite logits and positive temperature required");
  const shifted = logits.map((x) =>
    Math.exp((x - Math.max(...logits)) / temperature),
  );
  return shifted.map((x) => x / shifted.reduce((a, b) => a + b, 0));
}
export function filteredDistribution(
  logits,
  temperature = 1,
  topK = logits.length,
  topP = 1,
) {
  if (topK < 1 || topK > logits.length || topP <= 0 || topP > 1)
    throw new Error("Invalid truncation controls");
  const raw = softmax(logits, temperature);
  const order = raw
    .map((_, i) => i)
    .sort((a, b) => raw[b] - raw[a] || a - b)
    .slice(0, topK);
  const denominator = order.reduce((a, i) => a + raw[i], 0);
  let cumulative = 0;
  const kept = [];
  for (const i of order) {
    kept.push(i);
    cumulative += raw[i] / denominator;
    if (cumulative >= topP) break;
  }
  const total = kept.reduce((a, i) => a + raw[i], 0);
  return raw.map((value, i) => (kept.includes(i) ? value / total : 0));
}
export function attentionRows(mode = "causal") {
  const q = [
      [1, 0],
      [0, 1],
      [1, 1],
    ],
    values = [
      [1, 0],
      [0, 2],
      [3, 1],
    ],
    segments = [0, 0, 1];
  const weights = q.map((query, i) => {
    const allowed = q
      .map((_, j) => j)
      .filter(
        (j) =>
          (mode === "bidirectional" || j <= i) &&
          (mode !== "packed" || segments[i] === segments[j]),
      );
    const scores = allowed.map(
      (j) =>
        query.reduce((sum, value, d) => sum + value * q[j][d], 0) /
        Math.sqrt(2),
    );
    const p = softmax(scores);
    return q.map((_, j) => (allowed.includes(j) ? p[allowed.indexOf(j)] : 0));
  });
  return {
    weights,
    outputs: weights.map((row) =>
      [0, 1].map((d) => row.reduce((sum, w, j) => sum + w * values[j][d], 0)),
    ),
    values,
  };
}
export function ropePair(vector, position, frequency = 1) {
  const t = position * frequency,
    [a, b] = vector;
  return [a * Math.cos(t) - b * Math.sin(t), a * Math.sin(t) + b * Math.cos(t)];
}
export function kvBytes(layers, batch, heads, positions, width, bytes = 2) {
  return 2 * layers * batch * heads * positions * width * bytes;
}
