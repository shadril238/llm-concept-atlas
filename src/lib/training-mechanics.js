export function crossEntropy(logits, target) {
  if (
    !logits.length ||
    !Number.isInteger(target) ||
    target < 0 ||
    target >= logits.length ||
    !logits.every(Number.isFinite)
  )
    throw Error("Finite logits and a valid integer target required.");
  const top = Math.max(...logits);
  return (
    top +
    Math.log(logits.reduce((s, x) => s + Math.exp(x - top), 0)) -
    logits[target]
  );
}
export function multipleNegatives(scores, scale = 20) {
  if (
    !Number.isFinite(scale) ||
    scale <= 0 ||
    scores.length < 2 ||
    scores.some((r) => r.length !== scores.length)
  )
    throw Error("Positive scale and a square paired matrix required.");
  const terms = scores.map((r, i) =>
    crossEntropy(
      r.map((x) => x * scale),
      i,
    ),
  );
  return { terms, loss: terms.reduce((a, b) => a + b, 0) / terms.length };
}
export function pairFeatures(a, b) {
  if (
    !a.length ||
    a.length !== b.length ||
    ![...a, ...b].every(Number.isFinite)
  )
    throw Error("Equal nonempty finite vectors required.");
  return [...a, ...b, ...a.map((x, i) => Math.abs(x - b[i]))];
}
export function cosineRegression(scores, targets) {
  if (
    !scores.length ||
    scores.length !== targets.length ||
    ![...scores, ...targets].every(Number.isFinite)
  )
    throw Error("Matching finite scores and targets required.");
  const terms = scores.map((v, i) => (v - targets[i]) ** 2);
  return { terms, loss: terms.reduce((a, b) => a + b, 0) / terms.length };
}
export function supervisedMask(tokens, selected) {
  if (
    new Set(selected).size !== selected.length ||
    selected.some((i) => !Number.isInteger(i) || i < 0 || i >= tokens.length)
  )
    throw Error("Unique valid token positions required.");
  if (selected.some((i) => tokens[i].special_or_padding))
    throw Error(
      "Special and padding tokens cannot be supervised in this MLM exercise.",
    );
  return tokens.map((t, i) => (selected.includes(i) ? t.original_id : -100));
}
