import { useState } from "react";
import recorded from "../../data/discovery-results.json";
import {
  classificationMetrics,
  parseLabel,
  classTfidf,
  mmr,
} from "../../lib/discovery-mechanics.js";

type Prediction = {
  id: string;
  prediction: number;
  confidence: number | null;
  scores: number[] | null;
  raw?: string;
};
type Review = {
  id: string;
  text: string;
  label: number;
  split: string;
  group?: string;
};
type Method = {
  contract: string;
  cases: Prediction[];
  timing: { median_seconds: number };
  test: { macro_f1: number; accuracy: number; invalid_rate: number };
};
type Paper = {
  id: string;
  title: string;
  text: string;
  topic: number;
  changed_topic: number;
  membership_strength: number;
  display_coordinates: number[];
};
type Topic = { id: number; count: number; representative_ids: string[] };
type TopicData = {
  documents: Paper[];
  topics: Topic[];
  representations: Record<
    string,
    Record<string, { term: string; weight: number }[]>
  >;
  topic_similarity: { topic_ids: number[]; cosines: number[][] };
  hierarchy: { topic_ids: number[]; linkage: number[][] };
  density_comparison: {
    ari_including_noise: number;
    before_noise: number;
    after_noise: number;
    contingency: { before: number; after: number; count: number }[];
  };
  reload_inference: { agreement_with_original_training_assignments: number };
};
const data = recorded as unknown as {
  classification: {
    rows: Review[];
    methods: Record<string, Method>;
    selection: { review_threshold: number };
    wording: { changed_ids: string[] };
  };
  topics: TopicData;
  generation?: {
    labels: {
      topic: number;
      label: string;
      prompt: string;
      review_status: string;
    }[];
  };
};
const methodNames: Record<string, string> = {
  "tfidf-logistic": "TF-IDF + logistic",
  "embedding-logistic": "Embeddings + logistic",
  "class-prototypes": "Class-average vectors",
  "description-similarity": "Label-description similarity",
  "task-encoder": "Task-specific encoder",
  "nli-zero-shot": "NLI zero-shot",
  "flan-t5": "FLAN-T5 generation",
};
const name = (id: number) =>
  id === 0 ? "negative" : id === 1 ? "positive" : "invalid / rejected";
const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

export function ClassificationComparison({
  initialMethod = "task-encoder",
}: {
  initialMethod?: string;
}) {
  const [method, setMethod] = useState(initialMethod),
    [split, setSplit] = useState("test"),
    [cell, setCell] = useState<number[] | null>(null),
    [threshold, setThreshold] = useState(
      data.classification.selection.review_threshold,
    );
  const chosen = data.classification.methods[method];
  const rows = data.classification.rows.filter((row) => row.split === split);
  const cases = rows.map((row) => ({
    ...row,
    ...chosen.cases.find((p) => p.id === row.id)!,
  }));
  const metrics = classificationMetrics(
    cases.map((r) => r.label),
    cases.map((r) => r.prediction),
  );
  const filtered = cell
    ? cases.filter(
        (r) =>
          r.label === cell[0] &&
          (r.prediction < 0 ? 2 : r.prediction) === cell[1],
      )
    : cases;
  const reviewCases = rows.map((row) => ({
    ...row,
    ...data.classification.methods["embedding-logistic"].cases.find(
      (p) => p.id === row.id,
    )!,
  }));
  const accepted = reviewCases.filter(
    (row) => row.confidence !== null && row.confidence >= threshold,
  );
  const accurate = accepted.filter(
    (row) => row.label === row.prediction,
  ).length;
  return (
    <section
      className="reference-lab discovery-lab"
      aria-label="Classification comparison experiment"
    >
      <div className="reference-top">
        <span className="reference-label">
          Recorded model outputs · calculated metrics
        </span>
        <button
          onClick={() => {
            setMethod(initialMethod);
            setSplit("test");
            setCell(null);
            setThreshold(data.classification.selection.review_threshold);
          }}
        >
          Reset
        </button>
      </div>
      <h3>Same reviews. Different decisions.</h3>
      <p>
        Choose a method, then click a confusion-matrix cell to inspect its
        actual cases. Changing these controls selects recorded predictions; it
        does not rerun a model.
      </p>
      <div className="reference-controls">
        <label>
          Classification method
          <select
            value={method}
            onChange={(e) => {
              setMethod(e.target.value);
              setCell(null);
            }}
          >
            {Object.entries(methodNames).map(([key, label]) => (
              <option key={key} value={key}>
                {label}
              </option>
            ))}
          </select>
        </label>
        <label>
          Evaluation slice
          <select
            value={split}
            onChange={(e) => {
              setSplit(e.target.value);
              setCell(null);
            }}
          >
            <option value="test">80 UCI movie sentences</option>
            <option value="challenge">12 original transfer probes</option>
          </select>
        </label>
      </div>
      <p className="discovery-contract">{chosen.contract}</p>
      <div className="discovery-stats">
        <div>
          <strong>{metrics.macro_f1.toFixed(3)}</strong>
          <span>Macro-F1 · two fixed labels</span>
        </div>
        <div>
          <strong>{pct(metrics.accuracy)}</strong>
          <span>Accuracy · {cases.length} cases</span>
        </div>
        <div>
          <strong>{pct(metrics.invalid_rate)}</strong>
          <span>Invalid output rate</span>
        </div>
      </div>
      <div className="reference-table">
        <table className="discovery-confusion">
          <caption>
            Rows = actual label · columns = predicted label. Invalid outputs
            remain in recall and accuracy denominators.
          </caption>
          <thead>
            <tr>
              <th scope="col">Actual / predicted</th>
              {["Negative", "Positive", "Invalid"].map((t) => (
                <th scope="col" key={t}>
                  {t}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {metrics.confusion.map((row: number[], actual: number) => (
              <tr key={actual}>
                <th scope="row">{name(actual)}</th>
                {row.map((count: number, pred: number) => (
                  <td key={pred}>
                    <button
                      aria-label={`Actual ${name(actual)}, predicted ${pred === 2 ? "invalid" : name(pred)}: ${count} cases`}
                      aria-pressed={cell?.[0] === actual && cell?.[1] === pred}
                      onClick={() => setCell([actual, pred])}
                      style={{
                        background: `color-mix(in srgb, var(--sl-color-text-accent) ${Math.min(65, 10 + count * 1.3)}%, var(--sl-color-black))`,
                      }}
                    >
                      {count}
                    </button>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="reference-table">
        <table>
          <caption>
            Per-class metrics · undefined precision/recall convention = 0
          </caption>
          <thead>
            <tr>
              <th>Class</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1</th>
              <th>Support</th>
            </tr>
          </thead>
          <tbody>
            {metrics.per_class.map(
              (row: {
                label: number;
                precision: number;
                recall: number;
                f1: number;
                support: number;
              }) => (
                <tr key={row.label}>
                  <th>{name(row.label)}</th>
                  <td>{row.precision.toFixed(3)}</td>
                  <td>{row.recall.toFixed(3)}</td>
                  <td>{row.f1.toFixed(3)}</td>
                  <td>{row.support}</td>
                </tr>
              ),
            )}
          </tbody>
        </table>
      </div>
      <details open={!!cell}>
        <summary>
          {cell
            ? `${filtered.length} cases in selected cell`
            : "Inspect all cases and model disagreements"}
        </summary>
        <button onClick={() => setCell(null)}>Show all cases</button>
        <ol className="discovery-cases">
          {filtered.map((row) => (
            <li key={row.id}>
              <small>
                {row.id}
                {row.group ? ` · ${row.group}` : ""}
              </small>
              <p>{row.text}</p>
              <strong>
                Actual: {name(row.label)} · predicted: {name(row.prediction)}
              </strong>
              {row.raw !== undefined && (
                <p>
                  Raw generated text: <code>{JSON.stringify(row.raw)}</code>
                </p>
              )}
              {row.scores && (
                <p>
                  Scores [negative, positive]:{" "}
                  {row.scores.map((v) => v.toFixed(3)).join(", ")}
                </p>
              )}
              <p className="discovery-disagreements">
                {Object.entries(data.classification.methods).map(
                  ([key, value]) => (
                    <span key={key}>
                      {methodNames[key]}:{" "}
                      <b>
                        {name(
                          value.cases.find((p) => p.id === row.id)!.prediction,
                        )}
                      </b>
                    </span>
                  ),
                )}
              </p>
            </li>
          ))}
        </ol>
        {!filtered.length && (
          <p>No cases in this cell in the recorded slice.</p>
        )}
      </details>
      <details>
        <summary>Compare all seven methods on the fixed test set</summary>
        <div className="reference-table">
          <table>
            <caption>
              Exploratory CPU comparison; prior training differs. Timing is
              median of three warmed passes over 92 held-out/challenge rows,
              excluding model loading.
            </caption>
            <thead>
              <tr>
                <th>Method</th>
                <th>Macro-F1</th>
                <th>Accuracy</th>
                <th>Invalid</th>
                <th>CPU seconds</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(data.classification.methods).map(
                ([key, value]) => (
                  <tr key={key}>
                    <th>{methodNames[key]}</th>
                    <td>{value.test.macro_f1.toFixed(3)}</td>
                    <td>{pct(value.test.accuracy)}</td>
                    <td>{pct(value.test.invalid_rate)}</td>
                    <td>{value.timing.median_seconds.toFixed(3)}</td>
                  </tr>
                ),
              )}
            </tbody>
          </table>
        </div>
        <p>
          Changing only the label-description wording changed{" "}
          {data.classification.wording.changed_ids.length} of 92 predictions.
          Similarity scores do not become probabilities because the largest one
          wins.
        </p>
      </details>
      <h4>Explore a human-review threshold</h4>
      <p>
        This control always uses the <strong>embedding + logistic</strong>{" "}
        model. The recorded validation-selected threshold is{" "}
        {data.classification.selection.review_threshold.toFixed(2)}. Moving it
        here explores a tradeoff; test cases must not be used to choose a
        deployment threshold.
      </p>
      <label>
        Review threshold: {threshold.toFixed(2)}
        <input
          type="range"
          min="0.5"
          max="1"
          step="0.01"
          value={threshold}
          onChange={(e) => setThreshold(Number(e.target.value))}
        />
      </label>
      <div className="reference-output" aria-live="polite">
        <strong>
          {accepted.length}/{rows.length} accepted ·{" "}
          {rows.length - accepted.length} for review
        </strong>
        <p>
          Coverage {pct(accepted.length / rows.length)}. Accuracy among accepted
          cases:{" "}
          {accepted.length
            ? pct(accurate / accepted.length)
            : "undefined: none accepted"}
          . Scores are model outputs, not guarantees of correctness. Reviewing
          every case does not produce 100% accepted accuracy.
        </p>
      </div>
    </section>
  );
}

export function LabelValidator() {
  const [raw, setRaw] = useState("positive");
  const label = parseLabel(raw);
  return (
    <section
      className="reference-lab discovery-lab"
      aria-label="Generated label validator"
    >
      <span className="reference-label">
        Calculated · exact output contract
      </span>
      <h3>A label is a contract</h3>
      <p>
        Try “positive”, “Negative”, “not negative”, or “positive because…”. Only
        whitespace trimming and case folding are allowed; unfamiliar output is
        never silently mapped to positive.
      </p>
      <label>
        Generated label text
        <input
          type="text"
          value={raw}
          onChange={(e) => setRaw(e.target.value)}
        />
      </label>
      <div className="reference-output" aria-live="polite">
        <strong>
          {label < 0 ? "Invalid → -1" : `${name(label)} → ${label}`}
        </strong>
        <p>
          An invalid output remains an error in the full evaluation and can be
          sent for review.
        </p>
      </div>
    </section>
  );
}

export function CtfidfCalculator() {
  const [counts, setCounts] = useState([
      [3, 1, 0],
      [0, 1, 3],
    ]),
    [word, setWord] = useState(0);
  let result: ReturnType<typeof classTfidf> | null = null;
  try {
    result = classTfidf(counts);
  } catch {
    /* Show the invalid table instead of silently fixing counts. */
  }
  const words = ["translation", "model", "speech"];
  return (
    <section
      className="reference-lab discovery-lab"
      aria-label="Class-based TF-IDF calculator"
    >
      <div className="reference-top">
        <span className="reference-label">
          Calculated · two invented class documents
        </span>
        <button
          onClick={() => {
            setCounts([
              [3, 1, 0],
              [0, 1, 3],
            ]);
            setWord(0);
          }}
        >
          Reset
        </button>
      </div>
      <h3>Why a common word loses weight</h3>
      <p>
        Change a count, predict its effect, then inspect the exact factors.
        These groups are fixed only for this arithmetic example; the connected
        BERTopic lab fits its own assignments.
      </p>
      <div className="reference-table">
        <table>
          <caption>
            Token counts after aggregating documents within each class
          </caption>
          <thead>
            <tr>
              <th>Class</th>
              {words.map((term) => (
                <th key={term}>{term}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {counts.map((row, i) => (
              <tr key={i}>
                <th>{i}</th>
                {row.map((v, j) => (
                  <td key={j}>
                    <input
                      aria-label={`Class ${i} ${words[j]} count`}
                      type="number"
                      min="0"
                      max="20"
                      step="1"
                      value={v}
                      onChange={(e) => {
                        const value = Number(e.target.value);
                        if (
                          Number.isInteger(value) &&
                          value >= 0 &&
                          value <= 20
                        )
                          setCounts(
                            counts.map((r, k) =>
                              k === i
                                ? r.map((n, l) => (l === j ? value : n))
                                : r,
                            ),
                          );
                      }}
                    />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <label>
        Inspect term
        <select value={word} onChange={(e) => setWord(Number(e.target.value))}>
          {words.map((term, i) => (
            <option key={term} value={i}>
              {term}
            </option>
          ))}
        </select>
      </label>
      {result ? (
        <div className="reference-output" aria-live="polite">
          <p>
            A = floor(mean class token count) ={" "}
            <strong>{result.average_length}</strong>. Total frequency of “
            {words[word]}” across classes ={" "}
            <strong>{result.frequencies[word]}</strong>.
          </p>
          <p>
            IDF = ln(1 + {result.average_length}/{result.frequencies[word]}) ={" "}
            <strong>{result.idf[word].toFixed(6)}</strong>.
          </p>
          {counts.map((row, i) => (
            <p key={i}>
              Class {i}: TF = {row[word]}/{row.reduce((a, b) => a + b, 0)} ={" "}
              {result.tf[i][word].toFixed(6)}; weight ={" "}
              <strong>{result.weights[i][word].toFixed(6)}</strong>.
            </p>
          ))}
          <p>
            Default BERTopic 0.17.4 convention: L1 class term frequencies × ln(1
            + A/f). Weights are not topic probabilities.
          </p>
        </div>
      ) : (
        <p role="status">
          Each class and each vocabulary column must contain a token. Restore a
          positive count or reset the table.
        </p>
      )}
    </section>
  );
}

export function DiversityExplorer() {
  const [diversity, setDiversity] = useState(0.3);
  const terms = ["translation", "translations", "language", "evaluation"],
    relevance = [0.9, 0.88, 0.7, 0.65],
    similarities = [
      [1, 0.98, 0.55, 0.2],
      [0.98, 1, 0.54, 0.2],
      [0.55, 0.54, 1, 0.3],
      [0.2, 0.2, 0.3, 1],
    ];
  const selected = mmr(terms, relevance, similarities, diversity, 3);
  return (
    <section
      className="reference-lab discovery-lab"
      aria-label="MMR diversity experiment"
    >
      <span className="reference-label">
        Calculated · illustrative similarities
      </span>
      <h3>Choose useful variety</h3>
      <p>
        Predict whether “translations” survives as diversity grows. MMR first
        chooses the most relevant candidate; later choices balance relevance
        against similarity to already selected terms.
      </p>
      <label>
        Diversity: {diversity.toFixed(2)}
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={diversity}
          onChange={(e) => setDiversity(Number(e.target.value))}
        />
      </label>
      <div className="reference-output" aria-live="polite">
        <strong>Selection order: {selected.join(" → ")}</strong>
        <p>
          Next score = (1 − diversity) × relevance − diversity × maximum
          similarity to a selected term. At diversity 1, later choices ignore
          relevance, so more variety can become less useful.
        </p>
      </div>
      <div className="reference-table">
        <table>
          <caption>
            Invented fixed relevance and pair similarities, not live encoder
            output
          </caption>
          <thead>
            <tr>
              <th>Candidate</th>
              <th>Relevance</th>
              {terms.map((t) => (
                <th key={t}>{t}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {terms.map((t, i) => (
              <tr key={t}>
                <th>{t}</th>
                <td>{relevance[i]}</td>
                {similarities[i].map((v, j) => (
                  <td key={j}>{v}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

const colors = ["#27a89e", "#bb8de0", "#da9f35", "#6698ce"];
const topicColor = (topic: number) =>
  topic < 0 ? "#8a8e97" : colors[topic % colors.length];
export function TopicExplorer() {
  const [representation, setRepresentation] = useState("ctfidf"),
    [topic, setTopic] = useState("all"),
    [view, setView] = useState("map"),
    [density, setDensity] = useState("baseline"),
    [selected, setSelected] = useState(data.topics.documents[0].id);
  const t = data.topics;
  const visible = t.documents.filter(
    (d) =>
      topic === "all" ||
      (density === "baseline" ? d.topic : d.changed_topic) === Number(topic),
  );
  const active =
    visible.find((d) => d.id === selected) ??
    visible[0] ??
    t.documents.find((d) => d.id === selected)!;
  const baselineTopic = topic === "all" ? active.topic : Number(topic);
  const terms =
    t.representations[representation]?.[String(baselineTopic)] ?? [];
  const representativeIds =
    t.topics.find((row) => row.id === active.topic)?.representative_ids ?? [];
  const xs = t.documents.map((d) => d.display_coordinates[0]),
    ys = t.documents.map((d) => d.display_coordinates[1]);
  const x = (v: number) =>
    35 +
    ((v - Math.min(...xs)) / (Math.max(...xs) - Math.min(...xs) || 1)) * 560;
  const y = (v: number) =>
    275 -
    ((v - Math.min(...ys)) / (Math.max(...ys) - Math.min(...ys) || 1)) * 235;
  return (
    <section
      className="reference-lab discovery-lab"
      aria-label="Research collection topic explorer"
    >
      <div className="reference-top">
        <span className="reference-label">
          Recorded fitted BERTopic · original fictional summaries
        </span>
        <button
          onClick={() => {
            setRepresentation("ctfidf");
            setTopic("all");
            setView("map");
            setDensity("baseline");
            setSelected(t.documents[0].id);
          }}
        >
          Reset
        </button>
      </div>
      <h3>Explore the documents behind a topic</h3>
      <p>
        45 texts → [45,384] embeddings → [45,5] UMAP coordinates → HDBSCAN
        groups → topic representations. The map uses an independent [45,2]
        display projection. These controls inspect recorded runs.
      </p>
      <div className="reference-controls">
        <label>
          Explorer view
          <select value={view} onChange={(e) => setView(e.target.value)}>
            <option value="map">Document map and evidence</option>
            <option value="terms">Topic words and generated labels</option>
            <option value="similarity">Topic similarity heatmap</option>
            <option value="hierarchy">Topic hierarchy</option>
            <option value="changes">Density change experiment</option>
          </select>
        </label>
        <label>
          Topic filter
          <select value={topic} onChange={(e) => setTopic(e.target.value)}>
            <option value="all">All documents</option>
            {t.topics.map((row) => (
              <option key={row.id} value={row.id}>
                {row.id < 0 ? "−1 · outliers" : `Topic ${row.id}`}
              </option>
            ))}
          </select>
        </label>
      </div>
      {view === "map" && (
        <>
          <label>
            Recorded density setting
            <select
              value={density}
              onChange={(e) => setDensity(e.target.value)}
            >
              <option value="baseline">min_samples 8 · baseline</option>
              <option value="changed">min_samples 3 · changed</option>
            </select>
          </label>
          <svg
            className="discovery-map"
            viewBox="0 0 630 310"
            role="img"
            aria-label="Two-dimensional display projection. Inspect individual documents in the equivalent selectable list below."
          >
            <rect
              x="15"
              y="15"
              width="600"
              height="280"
              rx="12"
              fill="var(--sl-color-black)"
            />
            {visible.map((d) => {
              const label = density === "baseline" ? d.topic : d.changed_topic,
                px = x(d.display_coordinates[0]),
                py = y(d.display_coordinates[1]);
              return (
                <g
                  key={d.id}
                  onClick={() => setSelected(d.id)}
                  style={{ cursor: "pointer" }}
                >
                  <title>
                    {d.id}: {d.title}; topic {label}
                  </title>
                  {label < 0 ? (
                    <path
                      d={`M ${px} ${py - 7} l 7 7 l -7 7 l -7 -7 Z`}
                      fill={topicColor(label)}
                      stroke={
                        active.id === d.id ? "var(--sl-color-white)" : "none"
                      }
                      strokeWidth="3"
                    />
                  ) : (
                    <circle
                      cx={px}
                      cy={py}
                      r={active.id === d.id ? 8 : 5.5}
                      fill={topicColor(label)}
                      stroke={
                        active.id === d.id ? "var(--sl-color-white)" : "none"
                      }
                      strokeWidth="3"
                    />
                  )}
                </g>
              );
            })}
          </svg>
          <p className="discovery-legend">
            {t.topics.map((row) => (
              <span key={row.id}>
                <i style={{ background: topicColor(row.id) }} />
                {row.id < 0 ? "◆ outliers" : `● topic ${row.id}`}
              </span>
            ))}
          </p>
          <p>
            No apparent gap, distance, or global orientation in this plot proves
            semantic separation. The same display points are colored by the two
            fitted partitions; numeric topic IDs may be permuted.
          </p>
          <label>
            Inspect document
            <select
              value={visible.length ? active.id : ""}
              disabled={!visible.length}
              onChange={(e) => setSelected(e.target.value)}
            >
              {!visible.length && (
                <option value="">No matching documents</option>
              )}
              {visible.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.id} · {d.title}
                </option>
              ))}
            </select>
          </label>
          {visible.length > 0 ? (
            <div className="reference-output">
              <small>{active.id} · original fictional research summary</small>
              <h4>{active.title}</h4>
              <p>{active.text}</p>
              <p>
                Baseline topic {active.topic}; density-change topic{" "}
                {active.changed_topic}. Baseline membership strength{" "}
                {active.membership_strength.toFixed(3)}. This density quantity
                is not semantic correctness.
              </p>
              <p>
                {representativeIds.includes(active.id)
                  ? "Selected as a representative document for the baseline group."
                  : "Not selected among this baseline group’s representative documents."}
              </p>
              <p>
                Representatives:{" "}
                {representativeIds.join(", ") || "none recorded"}. Read minority
                and unrelated documents too: a representative can hide the
                group’s weaknesses.
              </p>
            </div>
          ) : (
            <p role="status">
              No documents match this topic under the selected density setting.
              Choose another topic or all documents.
            </p>
          )}
          <details>
            <summary>
              Selectable document list · {visible.length} matching documents
            </summary>
            <ul className="discovery-document-list">
              {visible.map((d) => (
                <li key={d.id}>
                  <button
                    aria-pressed={active.id === d.id}
                    onClick={() => setSelected(d.id)}
                  >
                    {d.id} · {d.title}
                  </button>
                </li>
              ))}
            </ul>
          </details>
        </>
      )}
      {view === "terms" && (
        <>
          <p>
            Representations below always describe{" "}
            <strong>baseline assignments</strong>. Changing the representation
            does not move any document. Select a topic or inspect the current
            document’s baseline topic.
          </p>
          <label>
            Representation module
            <select
              value={representation}
              onChange={(e) => setRepresentation(e.target.value)}
            >
              <option value="ctfidf">Default c-TF-IDF</option>
              <option value="keybert">KeyBERTInspired</option>
              <option value="mmr-0.2">MMR diversity 0.2</option>
              <option value="mmr-0.8">MMR diversity 0.8</option>
            </select>
          </label>
          <h4>
            Baseline topic {baselineTopic}
            {baselineTopic < 0 ? " · noise is not a coherent topic" : ""}
          </h4>
          {terms.map((row) => (
            <div className="discovery-term" key={row.term}>
              <span>{row.term || "empty"}</span>
              <meter
                min="0"
                max={Math.max(...terms.map((r) => r.weight), 0.01)}
                value={Math.max(row.weight, 0)}
              />
              <code>{row.weight.toFixed(4)}</code>
            </div>
          ))}
          <p>
            {representation === "keybert"
              ? "Weights are term-to-representative semantic similarities."
              : "Weights are c-TF-IDF values; MMR selects a diverse subset and retains those term weights."}{" "}
            Compare term membership across representations, not absolute scores
            with different meanings.
          </p>
          {data.generation?.labels.find((r) => r.topic === baselineTopic) && (
            <div className="reference-output">
              <strong>
                Local FLAN-T5 name:{" "}
                {
                  data.generation.labels.find((r) => r.topic === baselineTopic)!
                    .label
                }
              </strong>
              <p>
                Executed generation; a human must check support and coverage. A
                fluent name does not verify the documents or create a better
                partition.
              </p>
              <details>
                <summary>Inspect the actual generation prompt</summary>
                <pre className="reference-input">
                  {
                    data.generation.labels.find(
                      (r) => r.topic === baselineTopic,
                    )!.prompt
                  }
                </pre>
              </details>
            </div>
          )}
          <p>
            <strong>Review task:</strong> Find a term or generated name that
            overlooks a minority document. Check its full text in the map view.
          </p>
        </>
      )}
      {view === "similarity" && (
        <div className="reference-table">
          <table>
            <caption>
              Cosine similarity of baseline fitted topic embeddings · includes
              −1 for transparency. Values are not evidence of topic accuracy.
            </caption>
            <thead>
              <tr>
                <th>Topic</th>
                {t.topic_similarity.topic_ids.map((id) => (
                  <th key={id}>{id}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {t.topic_similarity.cosines.map((row, i) => (
                <tr key={i}>
                  <th>{t.topic_similarity.topic_ids[i]}</th>
                  {row.map((v, j) => (
                    <td
                      key={j}
                      style={{
                        background: `color-mix(in srgb, var(--sl-color-text-accent) ${Math.max(0, v) * 55}%, transparent)`,
                      }}
                    >
                      {v.toFixed(3)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {view === "hierarchy" && (
        <>
          <p>
            Average-linkage hierarchy from cosine distances between baseline
            topic vectors; outliers excluded. Leaf index order is [
            {t.hierarchy.topic_ids.join(", ")}]. A merge is a display summary,
            not a change to document assignments.
          </p>
          <ol>
            {t.hierarchy.linkage.map((row, i) => (
              <li key={i}>
                Merge nodes {row[0]} and {row[1]} → node{" "}
                {t.hierarchy.topic_ids.length + i}, distance {row[2].toFixed(3)}
                , {row[3]} topic leaves.
              </li>
            ))}
          </ol>
        </>
      )}
      {view === "changes" && (
        <>
          <div className="discovery-stats">
            <div>
              <strong>
                {t.density_comparison.before_noise} →{" "}
                {t.density_comparison.after_noise}
              </strong>
              <span>Outliers · min_samples 8 → 3</span>
            </div>
            <div>
              <strong>
                {t.density_comparison.ari_including_noise.toFixed(3)}
              </strong>
              <span>Adjusted Rand index · includes noise</span>
            </div>
          </div>
          <p>
            Only HDBSCAN min_samples changes. The 384D embeddings and 5D UMAP
            points are held fixed. ARI is invariant to renumbered topic IDs;
            grouping every −1 point together is an explicit convention.
          </p>
          <div className="reference-table">
            <table>
              <caption>Partition transition counts</caption>
              <thead>
                <tr>
                  <th>Baseline group</th>
                  <th>Changed group</th>
                  <th>Documents</th>
                </tr>
              </thead>
              <tbody>
                {t.density_comparison.contingency.map((row, i) => (
                  <tr key={i}>
                    <td>{row.before}</td>
                    <td>{row.after}</td>
                    <td>{row.count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p>
            The safetensors reload preserves topic data but uses cosine-to-topic
            inference. Its agreement with original training assignments is{" "}
            {pct(
              t.reload_inference.agreement_with_original_training_assignments,
            )}
            ; it is a different assignment rule from the fitted density model.
          </p>
        </>
      )}
    </section>
  );
}
