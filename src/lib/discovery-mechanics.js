// Same declared conventions as atlas_foundations.e4; cross-language tests guard drift.
export function classificationMetrics(gold, predicted) {
  if (
    !gold.length ||
    gold.length !== predicted.length ||
    gold.some((v) => ![0, 1].includes(v)) ||
    predicted.some((v) => ![-1, 0, 1].includes(v))
  )
    throw new Error("Aligned binary labels required");
  const confusion = [
    [0, 0, 0],
    [0, 0, 0],
  ];
  gold.forEach(
    (actual, i) => confusion[actual][predicted[i] < 0 ? 2 : predicted[i]]++,
  );
  const perClass = [0, 1].map((label) => {
    const tp = confusion[label][label],
      fp = confusion[1 - label][label],
      support = confusion[label].reduce((a, b) => a + b, 0),
      fn = support - tp;
    return {
      label,
      support,
      precision: tp + fp ? tp / (tp + fp) : 0,
      recall: support ? tp / support : 0,
      f1: 2 * tp + fp + fn ? (2 * tp) / (2 * tp + fp + fn) : 0,
    };
  });
  return {
    confusion,
    accuracy: (confusion[0][0] + confusion[1][1]) / gold.length,
    macro_f1: perClass.reduce((n, r) => n + r.f1, 0) / 2,
    weighted_f1:
      perClass.reduce((n, r) => n + r.f1 * r.support, 0) / gold.length,
    invalid_rate: (confusion[0][2] + confusion[1][2]) / gold.length,
    per_class: perClass,
  };
}
export function parseLabel(raw) {
  return typeof raw === "string"
    ? ({ negative: 0, positive: 1 }[raw.trim().toLowerCase()] ?? -1)
    : -1;
}
export function classTfidf(counts) {
  if (
    !counts.length ||
    !counts[0].length ||
    counts.some(
      (r) =>
        r.length !== counts[0].length ||
        r.some((v) => !Number.isFinite(v) || v < 0),
    )
  )
    throw new Error("Nonnegative rectangular counts required");
  const totals = counts.map((r) => r.reduce((a, b) => a + b, 0)),
    frequencies = counts[0].map((_, j) =>
      counts.reduce((n, row) => n + row[j], 0),
    );
  if (totals.some((v) => v <= 0) || frequencies.some((v) => v <= 0))
    throw new Error("Each class and vocabulary term must occur");
  const average_length = Math.floor(
      totals.reduce((a, b) => a + b, 0) / totals.length,
    ),
    idf = frequencies.map((f) => Math.log(1 + average_length / f)),
    tf = counts.map((r, i) => r.map((v) => v / totals[i]));
  return {
    average_length,
    frequencies,
    idf,
    tf,
    weights: tf.map((r) => r.map((v, j) => v * idf[j])),
  };
}
export function mmr(
  terms,
  relevance,
  similarities,
  diversity = 0.3,
  limit = 3,
) {
  const n = terms.length;
  if (
    !n ||
    new Set(terms).size !== n ||
    relevance.length !== n ||
    similarities.length !== n ||
    similarities.some((r) => r.length !== n) ||
    ![...relevance, ...similarities.flat(), diversity].every(Number.isFinite) ||
    diversity < 0 ||
    diversity > 1 ||
    !Number.isInteger(limit) ||
    limit < 1 ||
    limit > n
  )
    throw new Error("Invalid MMR inputs");
  const best = (candidates, score) =>
    candidates.reduce((a, b) => (score(b) > score(a) ? b : a));
  const chosen = [
    best(
      terms.map((_, i) => i),
      (i) => relevance[i],
    ),
  ];
  while (chosen.length < limit)
    chosen.push(
      best(
        terms.map((_, i) => i).filter((i) => !chosen.includes(i)),
        (i) =>
          (1 - diversity) * relevance[i] -
          diversity * Math.max(...chosen.map((j) => similarities[i][j])),
      ),
    );
  return chosen.map((i) => terms[i]);
}
