import { useState } from "react";
import {
  loraTrace,
  bioSpans,
  spanScore,
} from "../../lib/reference-mechanics.js";
import results from "../../data/reference-results.json";
const fixed = (value: number, digits = 2) =>
  (Math.round((value + Number.EPSILON) * 10 ** digits) / 10 ** digits).toFixed(
    digits,
  );

export function TokenizerExplorer() {
  const [model, setModel] = useState(0);
  const [sample, setSample] = useState(0);
  const [selected, setSelected] = useState<number | null>(null);
  const row = results.tokenizers[model * 4 + sample];
  return (
    <section
      className="reference-lab"
      aria-label="Recorded tokenizer comparison"
    >
      <div className="reference-top">
        <span className="reference-label">
          Recorded tokenizer output · CPU verified
        </span>
        <button
          onClick={() => {
            setModel(0);
            setSample(0);
            setSelected(null);
          }}
        >
          Reset
        </button>
      </div>
      <h3>Same text. Different vocabulary.</h3>
      <p>
        Predict the token count before switching tokenizer. Select a piece to
        inspect its ID and source offsets.
      </p>
      <div className="reference-controls">
        <label>
          Tokenizer
          <select
            value={model}
            onChange={(event) => {
              setModel(Number(event.target.value));
              setSelected(null);
            }}
          >
            <option value={0}>BERT uncased · WordPiece</option>
            <option value={1}>GPT-2 · byte BPE</option>
          </select>
        </label>
        <label>
          Input
          <select
            value={sample}
            onChange={(event) => {
              setSample(Number(event.target.value));
              setSelected(null);
            }}
          >
            {[
              "English sentence",
              "Unusual word",
              "Korean + emoji",
              "Python + whitespace",
            ].map((value, index) => (
              <option key={value} value={index}>
                {value}
              </option>
            ))}
          </select>
        </label>
      </div>
      <pre className="reference-input">{row.text}</pre>
      <div className="reference-tokens">
        {row.pieces.map((piece, index) => (
          <button
            key={index}
            aria-pressed={selected === index}
            className={row.special[index] ? "is-special" : ""}
            onClick={() => setSelected(index)}
          >
            {piece}
            <small>{row.ids[index]}</small>
          </button>
        ))}
      </div>
      <div className="reference-output" aria-live="polite">
        <strong>
          {row.ids.length} tokens, including{" "}
          {row.special.filter(Boolean).length} added markers
        </strong>
        <p>
          {selected === null
            ? "Choose a piece above."
            : `Piece ${selected}: ID ${row.ids[selected]}, source character offsets [${row.offsets[selected].join(", ")}). ${row.special[selected] ? "Special marker: no source characters." : "Byte pieces may share a character span."}`}
        </p>
        <p>
          Decoded: <code>{row.decoded}</code>
        </p>
      </div>
      <details>
        <summary>Text equivalent and interpretation</summary>
        <p>
          The IDs in order are {row.ids.join(", ")}. BERT normalizes case and
          may emit unknown tokens. GPT-2 preserves these inputs through its byte
          vocabulary. Visible byte symbols are tokenizer notation, not extra
          characters in the original text. These are fixed recorded cases; no
          model runs in this browser.
        </p>
      </details>
    </section>
  );
}

export function LoraExplorer() {
  const [alpha, setAlpha] = useState(1);
  const [rank, setRank] = useState(1);
  const [width, setWidth] = useState(512);
  const trace = loraTrace(alpha, rank);
  return (
    <section className="reference-lab" aria-label="Calculated LoRA trace">
      <div className="reference-top">
        <span className="reference-label">
          Calculated · fixed teaching matrices
        </span>
        <button
          onClick={() => {
            setAlpha(1);
            setRank(1);
            setWidth(512);
          }}
        >
          Reset
        </button>
      </div>
      <h3>Follow the low-rank branch</h3>
      <p>
        Predict what doubling alpha does. The trace uses one active factor; for
        r = 2 the extra factor is zero, so only the declared scaling changes.
      </p>
      <div className="reference-controls">
        <label>
          Alpha: {alpha}
          <input
            type="range"
            min={0}
            max={4}
            step={0.5}
            value={alpha}
            onChange={(e) => setAlpha(Number(e.target.value))}
          />
        </label>
        <label>
          Declared rank
          <select
            value={rank}
            onChange={(e) => setRank(Number(e.target.value))}
          >
            <option value={1}>1</option>
            <option value={2}>2 · second factor zero</option>
          </select>
        </label>
      </div>
      <div className="reference-pipeline">
        <div>
          <small>Compress · A x</small>
          <strong>{fixed(trace.compressed)}</strong>
          <span>A: r × 2</span>
        </div>
        <span aria-hidden="true">→</span>
        <div>
          <small>Expand · (α/r) B A x</small>
          <strong>[{trace.delta.map((x) => fixed(x)).join(", ")}]</strong>
          <span>B: 2 × r</span>
        </div>
        <span aria-hidden="true">→</span>
        <div>
          <small>Add frozen W x</small>
          <strong>[{trace.output.map((x) => fixed(x)).join(", ")}]</strong>
          <span>output: 2 × 1</span>
        </div>
      </div>
      <div className="reference-output" aria-live="polite">
        y = [2, 1] + [{trace.delta.map((x) => fixed(x, 3)).join(", ")}]. Alpha
        zero removes the update.
      </div>
      <label className="reference-width">
        Square layer width
        <select
          value={width}
          onChange={(e) => setWidth(Number(e.target.value))}
        >
          {[2, 512, 4096].map((n) => (
            <option key={n}>{n}</option>
          ))}
        </select>
      </label>
      <p>
        <strong>{(2 * rank * width).toLocaleString()}</strong> adapter
        parameters versus <strong>{(width * width).toLocaleString()}</strong>{" "}
        frozen base parameters ({((100 * 2 * rank) / width).toFixed(2)}%). This
        count changes width in a separate hypothetical layer; the numeric trace
        above stays 2 × 2.
      </p>
    </section>
  );
}

export function NerExplorer() {
  const [policy, setPolicy] = useState("first");
  const [mode, setMode] = useState("boundary");
  const tags = ["O", "B-PER", "I-PER", "O", "B-ORG", "I-ORG", "O"];
  const prediction =
    mode === "correct"
      ? tags
      : mode === "boundary"
        ? ["O", "B-PER", "I-PER", "O", "B-ORG", "O", "O"]
        : ["O", "B-PER", "I-PER", "O", "B-LOC", "I-LOC", "O"];
  const words = [
    "Yesterday",
    "Ada",
    "Lovelace",
    "joined",
    "Cedar",
    "Labs",
    ".",
  ];
  const metric = spanScore(tags, prediction);
  let previous: number | null = null;
  return (
    <section
      className="reference-lab"
      aria-label="Subword alignment and entity error experiment"
    >
      <div className="reference-top">
        <span className="reference-label">
          Recorded BERT pieces + calculated BIO policies
        </span>
        <button
          onClick={() => {
            setPolicy("first");
            setMode("boundary");
          }}
        >
          Reset
        </button>
      </div>
      <h3>Which pieces receive a training label?</h3>
      <p>
        The table uses the actual tokenizer output from the authored training
        sentence. Padding and special markers always have label −100.
      </p>
      <label>
        Label policy
        <select value={policy} onChange={(e) => setPolicy(e.target.value)}>
          <option value="first">First subtoken only</option>
          <option value="all">
            All subtokens · B becomes I on continuation
          </option>
        </select>
      </label>
      <div className="reference-table">
        <table>
          <thead>
            <tr>
              <th>Word index / word</th>
              <th>BIO</th>
              <th>Piece / ID</th>
              <th>Training label</th>
              <th>In loss?</th>
            </tr>
          </thead>
          <tbody>
            {results.ner.alignment.map((row, index) => {
              const continuation =
                row.wordId !== null && previous === row.wordId;
              previous = row.wordId;
              const label =
                row.wordId === null
                  ? "-100"
                  : policy === "first" && continuation
                    ? "-100"
                    : continuation
                      ? row.wordTag.replace(/^B-/, "I-")
                      : row.wordTag;
              return (
                <tr key={index}>
                  <td>
                    {row.wordId === null ? "—" : `${row.wordId} / ${row.word}`}
                  </td>
                  <td>{row.wordTag || "—"}</td>
                  <td>
                    <code>{row.piece}</code> / {row.id}
                  </td>
                  <td>
                    <code>{label}</code>
                  </td>
                  <td>{label === "-100" ? "Ignored" : "Yes"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <h3>A nearly correct tag sequence can miss an entity</h3>
      <label>
        Prediction error
        <select value={mode} onChange={(e) => setMode(e.target.value)}>
          <option value="correct">Correct spans</option>
          <option value="boundary">Cedar instead of Cedar Labs</option>
          <option value="type">Cedar Labs as LOC instead of ORG</option>
        </select>
      </label>
      <div className="reference-tokens">
        {words.map((word, index) => (
          <span key={index}>
            {word}
            <small>{prediction[index]}</small>
          </span>
        ))}
      </div>
      <div className="reference-output" aria-live="polite">
        <strong>
          Token accuracy:{" "}
          {(
            (100 * tags.filter((tag, i) => tag === prediction[i]).length) /
            tags.length
          ).toFixed(1)}
          % · strict entity F1: {metric.f1.toFixed(2)}
        </strong>
        <p>
          Gold: PER [1, 3), ORG [4, 6). Prediction:{" "}
          {bioSpans(prediction)
            .map((s) => `${s[0]} [${s[1]}, ${s[2]})`)
            .join("; ")}
          . Boundaries use word indices; the end is exclusive.
        </p>
      </div>
    </section>
  );
}
