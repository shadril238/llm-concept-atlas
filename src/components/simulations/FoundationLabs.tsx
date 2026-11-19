import { useState } from "react";
import {
  attentionRows,
  filteredDistribution,
  ropePair,
  kvBytes,
} from "../../lib/foundation-mechanics.js";
import data from "../../data/foundation-results.json";

export function AttentionStepper() {
  const [mode, setMode] = useState("causal");
  const [row, setRow] = useState(1);
  const trace = attentionRows(mode);
  return (
    <section className="reference-lab" aria-label="Attention matrix experiment">
      <div className="reference-top">
        <span className="reference-label">
          Calculated · three invented Q/K/V positions
        </span>
        <button
          onClick={() => {
            setMode("causal");
            setRow(1);
          }}
        >
          Reset
        </button>
      </div>
      <h3>Choose what a query can see</h3>
      <p>
        Predict which weights become zero before changing the mask. The same
        fixed Q/K/V vectors are used in the Python mechanism tests.
      </p>
      <div className="reference-controls">
        <label>
          Visibility
          <select value={mode} onChange={(e) => setMode(e.target.value)}>
            <option value="causal">Causal self-attention</option>
            <option value="bidirectional">Bidirectional self-attention</option>
            <option value="packed">
              Independent packed documents: [0, 0, 1]
            </option>
          </select>
        </label>
        <label>
          Inspect query
          <select value={row} onChange={(e) => setRow(Number(e.target.value))}>
            {[0, 1, 2].map((i) => (
              <option key={i} value={i}>
                Position {i}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="foundation-matrix">
        <table>
          <caption>Rows: query positions · columns: key positions</caption>
          <thead>
            <tr>
              <th>Q / K</th>
              {[0, 1, 2].map((i) => (
                <th key={i}>{i}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {trace.weights.map((weights, i) => (
              <tr key={i} className={row === i ? "selected-row" : ""}>
                <th>{i}</th>
                {weights.map((weight, j) => (
                  <td
                    key={j}
                    style={{
                      background:
                        weight === 0
                          ? "transparent"
                          : `color-mix(in srgb, var(--sl-color-text-accent) ${Math.round(weight * 55)}%, transparent)`,
                    }}
                  >
                    {weight === 0 ? "0 · blocked" : weight.toFixed(3)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="reference-output" aria-live="polite">
        <strong>
          Query {row}: [{trace.weights[row].map((x) => x.toFixed(3)).join(", ")}
          ]
        </strong>
        <p>
          Values are [1,0], [0,2], [3,1]. Their weighted mixture is [
          {trace.outputs[row].map((x) => x.toFixed(3)).join(", ")}]. Each
          permitted row sums to one. A separator token alone would not create
          the packed mask.
        </p>
      </div>
      <details>
        <summary>Read the computation without the heatmap</summary>
        <p>
          Scores are Q·K divided by √2. Forbidden positions are excluded before
          softmax. The weights above multiply values, not IDs. This one head is
          an illustration of a calculation, not evidence about a model's
          internal reasoning.
        </p>
      </details>
    </section>
  );
}

export function PositionExplorer() {
  const [qPos, setQPos] = useState(0),
    [kPos, setKPos] = useState(1),
    [frequency, setFrequency] = useState(0.5);
  const q = ropePair([1, 0], qPos, frequency),
    k = ropePair([1, 0], kPos, frequency);
  const dot = q[0] * k[0] + q[1] * k[1];
  return (
    <section className="reference-lab" aria-label="RoPE rotation experiment">
      <div className="reference-top">
        <span className="reference-label">
          Calculated · one coordinate pair
        </span>
        <button
          onClick={() => {
            setQPos(0);
            setKPos(1);
            setFrequency(0.5);
          }}
        >
          Reset
        </button>
      </div>
      <h3>Rotate queries and keys</h3>
      <p>
        Predict what happens if both positions move forward by one. The relative
        offset stays fixed; so does their dot product under this rotation
        convention.
      </p>
      <div className="reference-controls">
        <label>
          Query position: {qPos}
          <input
            type="range"
            min="0"
            max="10"
            value={qPos}
            onChange={(e) => setQPos(Number(e.target.value))}
          />
        </label>
        <label>
          Key position: {kPos}
          <input
            type="range"
            min="0"
            max="10"
            value={kPos}
            onChange={(e) => setKPos(Number(e.target.value))}
          />
        </label>
        <label>
          Radians per position
          <select
            value={frequency}
            onChange={(e) => setFrequency(Number(e.target.value))}
          >
            <option value={0.5}>0.5</option>
            <option value={1}>1.0</option>
          </select>
        </label>
      </div>
      <svg
        viewBox="0 0 320 260"
        className="foundation-rotation"
        role="img"
        aria-label={`Unit query vector at ${qPos * frequency} radians and unit key vector at ${kPos * frequency} radians`}
      >
        <circle
          cx="160"
          cy="130"
          r="90"
          fill="none"
          stroke="currentColor"
          opacity=".3"
        />
        <path d="M50 130H270M160 20V240" stroke="currentColor" opacity=".3" />
        <line
          x1="160"
          y1="130"
          x2={160 + q[0] * 90}
          y2={130 - q[1] * 90}
          stroke="var(--sl-color-text-accent)"
          strokeWidth="4"
        />
        <circle
          cx={160 + q[0] * 90}
          cy={130 - q[1] * 90}
          r="6"
          fill="var(--sl-color-text-accent)"
        />
        <line
          x1="160"
          y1="130"
          x2={160 + k[0] * 90}
          y2={130 - k[1] * 90}
          stroke="#d48ff4"
          strokeWidth="3"
          strokeDasharray="6 4"
        />
        <rect
          x={156 + k[0] * 90}
          y={126 - k[1] * 90}
          width="8"
          height="8"
          fill="#d48ff4"
        />
      </svg>
      <div className="reference-output" aria-live="polite">
        Query (solid circle): [{q.map((x) => x.toFixed(3)).join(", ")}]. Key
        (dashed square): [{k.map((x) => x.toFixed(3)).join(", ")}]. Dot product:{" "}
        <strong>{dot.toFixed(4)}</strong>. Both norms remain one. Real models
        use many coordinate pairs/frequencies.
      </div>
    </section>
  );
}

export function DecodingExplorer() {
  const [temperature, setTemperature] = useState(1),
    [k, setK] = useState(3),
    [p, setP] = useState(0.8);
  const values: number[] = filteredDistribution([2, 1, 0], temperature, k, p);
  return (
    <section
      className="reference-lab"
      aria-label="Decoding distribution experiment"
    >
      <div className="reference-top">
        <span className="reference-label">
          Calculated · fixed logits [2, 1, 0]
        </span>
        <button
          onClick={() => {
            setTemperature(1);
            setK(3);
            setP(0.8);
          }}
        >
          Reset
        </button>
      </div>
      <h3>Change the selection distribution</h3>
      <p>
        Predict whether top-p 0.8 keeps one, two, or three candidates. This
        implementation applies temperature, then top-k, then top-p on the
        retained mass.
      </p>
      <div className="reference-controls">
        <label>
          Temperature: {temperature.toFixed(1)}
          <input
            type="range"
            min=".1"
            max="2"
            step=".1"
            value={temperature}
            onChange={(e) => setTemperature(Number(e.target.value))}
          />
        </label>
        <label>
          Top-k
          <select value={k} onChange={(e) => setK(Number(e.target.value))}>
            {[1, 2, 3].map((i) => (
              <option key={i}>{i}</option>
            ))}
          </select>
        </label>
        <label>
          Top-p: {p.toFixed(2)}
          <input
            type="range"
            min=".1"
            max="1"
            step=".05"
            value={p}
            onChange={(e) => setP(Number(e.target.value))}
          />
        </label>
      </div>
      <div className="foundation-bars">
        {values.map((value, i) => (
          <div key={i}>
            <span>Candidate {i}</span>
            <meter
              min="0"
              max="1"
              value={value}
              aria-label={`Candidate ${i} probability`}
            />
            <strong>
              {value.toFixed(4)}
              {value === 0 ? " · removed" : ""}
            </strong>
          </div>
        ))}
      </div>
      <div className="reference-output" aria-live="polite">
        Sum: {values.reduce((a, b) => a + b, 0).toFixed(4)}. Greedy decoding
        chooses candidate 0 directly; it does not divide by temperature zero.
        These are illustrative logits, not model confidence scores.
      </div>
    </section>
  );
}

export function CacheExplorer() {
  const [positions, setPositions] = useState(128),
    [heads, setHeads] = useState(4),
    [bytes, setBytes] = useState(2);
  const size = kvBytes(6, 1, heads, positions, 32, bytes);
  return (
    <section className="reference-lab" aria-label="KV cache memory calculator">
      <div className="reference-top">
        <span className="reference-label">
          Calculated tensor estimate + recorded CPU check
        </span>
        <button
          onClick={() => {
            setPositions(128);
            setHeads(4);
            setBytes(2);
          }}
        >
          Reset
        </button>
      </div>
      <h3>Cache grows with retained positions</h3>
      <p>
        Six layers, batch one, head width 32. Predict the effect of halving KV
        heads or doubling context. Query-head count is not changed by this
        storage calculator.
      </p>
      <div className="reference-controls">
        <label>
          Positions: {positions}
          <input
            type="range"
            min="16"
            max="2048"
            step="16"
            value={positions}
            onChange={(e) => setPositions(Number(e.target.value))}
          />
        </label>
        <label>
          KV heads
          <select
            value={heads}
            onChange={(e) => setHeads(Number(e.target.value))}
          >
            {[1, 2, 4].map((n) => (
              <option key={n}>{n}</option>
            ))}
          </select>
        </label>
        <label>
          Bytes/value
          <select
            value={bytes}
            onChange={(e) => setBytes(Number(e.target.value))}
          >
            {[2, 4, 8].map((n) => (
              <option key={n}>{n}</option>
            ))}
          </select>
        </label>
      </div>
      <div className="reference-output" aria-live="polite">
        <strong>
          {size.toLocaleString()} bytes · {(size / 2 ** 20).toFixed(3)} MiB
        </strong>
        <p>
          2 × 6 × 1 × {heads} × {positions} × 32 × {bytes}. Counts K and V only;
          weights, activations, and allocator overhead are additional.
        </p>
      </div>
      <p>
        The actual CPU float64 check recorded cache shape [
        {data.cache.cache_shape.join(", ")}] per K/V layer and maximum
        cached/uncached logit error{" "}
        {data.cache.max_logit_error.toExponential(2)}. Changing these controls
        does not rerun the model.
      </p>
    </section>
  );
}

export function PlaylistExplorer() {
  const [index, setIndex] = useState(0);
  const row = data.playlists.test.cases[index];
  return (
    <section
      className="reference-lab"
      aria-label="Recorded playlist recommendations"
    >
      <div className="reference-top">
        <span className="reference-label">
          Recorded · learned vectors, held-out sessions
        </span>
        <button onClick={() => setIndex(0)}>Reset</button>
      </div>
      <h3>Recommend an unseen song</h3>
      <label>
        Held-out session
        <select
          value={index}
          onChange={(e) => setIndex(Number(e.target.value))}
        >
          {data.playlists.test.cases.map((item, i) => (
            <option key={item.id} value={i}>
              {item.id}
            </option>
          ))}
        </select>
      </label>
      <p>
        Observed: <strong>{row.observed.join(", ")}</strong>. Hidden target:{" "}
        <strong>{row.target}</strong>. Item prefixes label fictional themes;
        they were not provided as training labels.
      </p>
      <div className="foundation-comparison">
        <div>
          <h4>Learned cosine ranking</h4>
          <ol>
            {row.learned.map((song) => (
              <li key={song}>
                {song}
                {song === row.target ? " · target" : ""}
              </li>
            ))}
          </ol>
        </div>
        <div>
          <h4>Train-only popularity</h4>
          <ol>
            {row.popular.map((song) => (
              <li key={song}>
                {song}
                {song === row.target ? " · target" : ""}
              </li>
            ))}
          </ol>
        </div>
      </div>
      <div className="reference-output" aria-live="polite">
        Target rank: {row.rank}; popularity rank: {row.popular_rank}. Already
        observed songs are excluded. On {data.playlists.test.queries} test
        queries, learned Recall@3 is{" "}
        {data.playlists.test.recall_at_3.toFixed(3)} and popularity Recall@3 is{" "}
        {data.playlists.test.popularity_recall_at_3.toFixed(3)}.
      </div>
      <p>
        These original synthetic playlists test the workflow. They do not
        establish real-user preference quality. The cold-start item has no
        fitted interactions and is excluded from learned ranking.
      </p>
    </section>
  );
}

export function ArchitectureMap() {
  const [family, setFamily] = useState("decoder");
  const details =
    family === "encoder"
      ? [
          "Complete source IDs",
          "Bidirectional encoder states",
          "Pool or classify",
          "Vector / labels",
        ]
      : family === "seq2seq"
        ? [
            "Source → encoder memory",
            "Target prefix → causal decoder",
            "Cross-attend to source memory",
            "Next target-token scores",
          ]
        : [
            "Prefix IDs → embeddings",
            "Causal transformer blocks",
            "LM head → vocabulary logits",
            "Select next token → append",
          ];
  return (
    <section className="reference-lab" aria-label="Architecture to task map">
      <div className="reference-top">
        <span className="reference-label">
          Conceptual architecture map · no model call
        </span>
        <button onClick={() => setFamily("decoder")}>Reset</button>
      </div>
      <h3>Choose the input/output contract</h3>
      <label>
        Model family
        <select value={family} onChange={(e) => setFamily(e.target.value)}>
          <option value="encoder">Encoder-only · representation/tagging</option>
          <option value="decoder">Decoder-only · continuation</option>
          <option value="seq2seq">Encoder-decoder · source-to-target</option>
        </select>
      </label>
      <ol className="foundation-flow">
        {details.map((detail, i) => (
          <li key={detail}>
            <span>{i + 1}</span>
            {detail}
          </li>
        ))}
      </ol>
      <p aria-live="polite">
        {family === "encoder"
          ? "Known input positions can use both sides of context. A task-specific head or pooling policy determines the final output."
          : family === "seq2seq"
            ? "Decoder self-attention is causal over target positions; cross-attention reads a separate encoded source. Source and target lengths may differ."
            : "Available prefix positions can be processed together, but an unknown continuation is selected sequentially."}{" "}
        Architecture does not certify task quality.
      </p>
    </section>
  );
}

export function RepresentationExplorer() {
  const [space, setSpace] = useState("static"),
    [word, setWord] = useState<"king" | "bank" | "good">("bank");
  const projection = data.representations.projection;
  const range = Math.max(...projection.coordinates.flat().map(Math.abs), 0.01);
  return (
    <section
      className="reference-lab"
      aria-label="Recorded representation inspector"
    >
      <div className="reference-top">
        <span className="reference-label">
          Recorded · different representation contracts
        </span>
        <button
          onClick={() => {
            setSpace("static");
            setWord("bank");
          }}
        >
          Reset
        </button>
      </div>
      <h3>Lookup row, contextual state, or sentence?</h3>
      <label>
        Inspect representation
        <select value={space} onChange={(e) => setSpace(e.target.value)}>
          <option value="static">Pretrained static word vectors · 50D</option>
          <option value="context">Contextual bank states · 128D</option>
          <option value="sentence">Pooled sentence vectors · 384D</option>
        </select>
      </label>
      {space === "static" ? (
        <>
          <label>
            Neighbor query
            <select
              value={word}
              onChange={(e) => setWord(e.target.value as typeof word)}
            >
              {["king", "bank", "good"].map((value) => (
                <option key={value}>{value}</option>
              ))}
            </select>
          </label>
          <ol>
            {data.representations.neighbors[word].map((row) => (
              <li key={row.word}>
                {row.word}: cosine {row.cosine.toFixed(4)}
              </li>
            ))}
          </ol>
          <svg
            viewBox="0 0 400 240"
            role="img"
            aria-label="PCA projection of eight selected static word vectors, not original 50-dimensional distances"
          >
            {projection.coordinates.map((point, index) => {
              const x = 200 + (point[0] / range) * 145,
                y = 120 - (point[1] / range) * 85;
              return (
                <g key={index}>
                  <circle
                    cx={x}
                    cy={y}
                    r="4"
                    fill="var(--sl-color-text-accent)"
                  />
                  <text
                    x={x + 6}
                    y={y + (index % 2 ? 16 : -8)}
                    fontSize="11"
                    fill="currentColor"
                  >
                    {projection.words[index]}
                  </text>
                </g>
              );
            })}
          </svg>
          <p>
            PCA of eight selected rows retains{" "}
            {(projection.variance_fraction * 100).toFixed(1)}% of their centered
            variance in two dimensions. Neighbor scores above use the original
            50D vectors; this chart can distort neighborhoods.
          </p>
        </>
      ) : space === "context" ? (
        <div className="reference-output">
          <p>
            Same bank token ID: {data.representations.bank_token_id}. Initial
            lookup is exactly equal in both sentences.
          </p>
          <p>
            “The bank approved the loan.” versus “The bank flooded after rain.”
          </p>
          <strong>
            Final contextual cosine:{" "}
            {data.representations.bank_context_cosine.toFixed(4)}
          </strong>
          <p>
            Lookup shape [{data.representations.lookup_shape.join(", ")}]; final
            hidden shape [{data.representations.hidden_shape.join(", ")}]. The
            later states differ despite the shared initial row.
          </p>
        </div>
      ) : (
        <>
          <ol>
            {data.representations.sentences.map((text, i) => (
              <li key={i}>
                {i}: {text}
              </li>
            ))}
          </ol>
          <div className="reference-table">
            <table>
              <caption>
                Cosine matrix after mask-aware pooling and L2 normalization
              </caption>
              <tbody>
                {data.representations.sentence_cosines.map((row, i) => (
                  <tr key={i}>
                    <th>{i}</th>
                    {row.map((score, j) => (
                      <td key={j}>{score.toFixed(3)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p>
            These scores are model-specific similarity measures, not relevance
            probabilities. Shape: [
            {data.representations.sentence_shape.join(", ")}].
          </p>
        </>
      )}
    </section>
  );
}
