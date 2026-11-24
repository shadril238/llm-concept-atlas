/** Declared arithmetic only: no model training or inference in the browser. */
export function lowRank(dIn, dOut, rank, alpha) {
  if (
    ![dIn, dOut, rank, alpha].every(Number.isFinite) ||
    Math.min(dIn, dOut, rank) <= 0
  )
    throw new Error("Invalid dimensions");
  return {
    base: dIn * dOut,
    adapter: rank * (dIn + dOut),
    scale: alpha / rank,
  };
}
export function trainingMemory(
  parameters,
  trainable,
  bits = 32,
  activationMiB = 0,
) {
  if (
    parameters < 0 ||
    trainable < 0 ||
    trainable > parameters ||
    ![4, 16, 32].includes(bits)
  )
    throw new Error("Invalid memory profile");
  return [
    {
      name: "Base weights (idealized)",
      mib: (parameters * bits) / 8 / 2 ** 20,
    },
    { name: "Trainable parameters, fp32", mib: (trainable * 4) / 2 ** 20 },
    { name: "Gradients, fp32", mib: (trainable * 4) / 2 ** 20 },
    { name: "Adam moments, fp32", mib: (trainable * 8) / 2 ** 20 },
    { name: "Activations (your estimate)", mib: activationMiB },
  ];
}
export function preferenceLoss(pc, pr, rc, rr, beta) {
  if (![pc, pr, rc, rr, beta].every(Number.isFinite) || beta <= 0)
    throw new Error("Invalid DPO inputs");
  const margin = pc - pr - (rc - rr),
    z = beta * margin;
  return {
    margin,
    z,
    loss: Math.max(0, -z) + Math.log1p(Math.exp(-Math.abs(z))),
  };
}
export function lossMask(ids, prefix, padding = 0, mode = "completion") {
  if (!Number.isInteger(prefix) || prefix < 0 || prefix >= ids.length)
    throw new Error("No completion");
  return [
    ...ids.map((id, i) => (mode === "all" || i >= prefix ? id : -100)),
    ...Array(padding).fill(-100),
  ];
}
