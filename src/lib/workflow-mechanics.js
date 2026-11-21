/** Unique valid plurality; strict majority uses ALL samples, including invalids. */
export function aggregateAnswers(samples) {
  const counts = new Map();
  for (const value of samples)
    if (Number.isInteger(value))
      counts.set(value, (counts.get(value) ?? 0) + 1);
  const best = Math.max(0, ...counts.values());
  const leaders = [...counts]
    .filter(([, count]) => count === best)
    .map(([value]) => value);
  const winner = leaders.length === 1 ? leaders[0] : null;
  return {
    winner,
    invalid: samples.filter((v) => !Number.isInteger(v)).length,
    counts: Object.fromEntries(counts),
    strict_majority: winner !== null && best > samples.length / 2,
    total: samples.length,
  };
}
/** Strict order contract, including duplicate keys and integer lexical representation. */
export function validateOrder(raw) {
  try {
    const value = JSON.parse(raw);
    if (!value || Array.isArray(value) || typeof value !== "object")
      throw Error("Expected one object.");
    // Scan top-level object keys after JSON.parse establishes syntactic validity.
    const keys = [];
    let depth = 0;
    const tokens = [];
    for (let i = 0; i < raw.length; i++) {
      if (raw[i] === '"') {
        let end = i + 1;
        while (end < raw.length) {
          if (raw[end] === "\\") end += 2;
          else if (raw[end++] === '"') break;
        }
        let next = end;
        while (/\s/.test(raw[next] ?? "") && next < raw.length) next++;
        if (depth === 1 && raw[next] === ":") {
          keys.push(JSON.parse(raw.slice(i, end)));
          let start = next + 1;
          while (/\s/.test(raw[start] ?? "") && start < raw.length) start++;
          tokens.push(raw.slice(start).match(/^[^,}\s]+/)?.[0] ?? "");
        }
        i = end - 1;
      } else if (raw[i] === "{" || raw[i] === "[") depth++;
      else if (raw[i] === "}" || raw[i] === "]") depth--;
    }
    if (new Set(keys).size !== keys.length)
      throw Error("Duplicate keys are ambiguous.");
    if (
      keys.length !== 3 ||
      !["item", "quantity", "priority"].every((k) => keys.includes(k))
    )
      throw Error("Exactly item, quantity, priority are required.");
    if (!["pen", "book"].includes(value.item))
      throw Error("item must be pen or book.");
    if (!["normal", "urgent"].includes(value.priority))
      throw Error("priority must be normal or urgent.");
    if (
      !Number.isInteger(value.quantity) ||
      !/^[1-9]$/.test(tokens[keys.indexOf("quantity")] ?? "")
    )
      throw Error("quantity must be an integer from 1 through 9.");
    return { valid: true, value, error: null };
  } catch (error) {
    return {
      valid: false,
      value: null,
      error: error instanceof Error ? error.message : "Invalid JSON.",
    };
  }
}
