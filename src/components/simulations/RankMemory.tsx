import { useState } from "react";
import { lowRank, trainingMemory } from "../../lib/generation-mechanics.js";
const num = (n: number) => n.toLocaleString("en-US");

export function RankMemory() {
  const [rank, setRank] = useState(4),
    [alpha, setAlpha] = useState(8),
    [bits, setBits] = useState(32),
    [activation, setActivation] = useState(64);
  const layer = lowRank(128, 384, rank, alpha),
    trainable = layer.adapter * 6;
  const rows = trainingMemory(14067712, trainable, bits, activation),
    max = Math.max(...rows.map((r) => r.mib), 1);
  return (
    <section
      className="workflow-lab generation-lab"
      aria-label="Rank and training memory explorer"
    >
      <p className="workflow-eyebrow">
        Arithmetic model · not measured peak memory
      </p>
      <h3>Small adapters still need a base model</h3>
      <div className="workflow-controls">
        <label>
          Rank r: {rank}
          <input
            type="range"
            min="1"
            max="32"
            value={rank}
            onChange={(e) => setRank(+e.target.value)}
          />
        </label>
        <label>
          Alpha: {alpha}
          <input
            type="range"
            min="1"
            max="64"
            value={alpha}
            onChange={(e) => setAlpha(+e.target.value)}
          />
        </label>
        <label>
          Idealized base precision
          <select value={bits} onChange={(e) => setBits(+e.target.value)}>
            <option value="32">32 bit</option>
            <option value="16">16 bit</option>
            <option value="4">4 bit</option>
          </select>
        </label>
        <label>
          Activation estimate, MiB: {activation}
          <input
            type="range"
            min="0"
            max="512"
            step="8"
            value={activation}
            onChange={(e) => setActivation(+e.target.value)}
          />
        </label>
        <button
          onClick={() => {
            setRank(4);
            setAlpha(8);
            setBits(32);
            setActivation(64);
          }}
        >
          Reset memory
        </button>
      </div>
      <p aria-live="polite">
        Each fused QKV matrix is 384 × 128. A is {rank} × 128; B is 384 × {rank}
        . Across six layers: {num(trainable)} adapter parameters. Scale α/r ={" "}
        {layer.scale.toFixed(2)}. Increasing alpha changes scale, not parameter
        count.
      </p>
      <div className="generation-bars">
        {rows.map((r) => (
          <div key={r.name}>
            <span>
              {r.name}: {r.mib.toFixed(3)} MiB
            </span>
            <div
              role="img"
              aria-label={`${r.name}, ${r.mib.toFixed(3)} MiB`}
              style={{ width: `${Math.max(0.5, (r.mib / max) * 100)}%` }}
            />
          </div>
        ))}
      </div>
      <p>
        Idealized tensor subtotal:{" "}
        {rows.reduce((s, r) => s + r.mib, 0).toFixed(2)} MiB. Quantization
        scales/zeros, unquantized embeddings/head, padding, temporary buffers,
        framework overhead and optimizer bookkeeping are omitted. NF4 does not
        make every parameter four bits. KV cache is an inference term here;
        training activations are a different term. This is not a GPU capacity
        guarantee.
      </p>
    </section>
  );
}
