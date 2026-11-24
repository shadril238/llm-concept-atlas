import { useState } from "react";
import published from "../../data/training-results.json";
import {
  crossEntropy,
  pairFeatures,
  cosineRegression,
  supervisedMask,
} from "../../lib/training-mechanics.js";
const data: Record<string, any> = published;
const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
const fixed = (v: number | null) => (v === null ? "undefined" : v.toFixed(4));
const Json = ({ value }: { value: unknown }) => (
  <pre className="workflow-pre">{JSON.stringify(value, null, 2)}</pre>
);
const Stat = ({ label, value }: { label: string; value: string | number }) => (
  <div className="workflow-stat">
    <strong>{value}</strong>
    <span>{label}</span>
  </div>
);
function Curve({
  rows,
  x = "step",
  y,
  label,
  unit = "loss",
}: {
  rows: any[];
  x?: string;
  y: string;
  label: string;
  unit?: string;
}) {
  const points = rows.filter((r) => Number.isFinite(r[y]));
  if (!points.length) return <p>No curve applies to this condition.</p>;
  const xmax = Math.max(...points.map((r) => r[x]), 1),
    ymax =
      unit === "quality" ? 1 : Math.max(...points.map((r) => r[y]), 0.01) * 1.1;
  return (
    <figure className="training-curve">
      <svg viewBox="0 0 640 220" role="img" aria-label={label}>
        <title>{`${label}; exact values in the table below.`}</title>
        <line x1="55" y1="175" x2="615" y2="175" />
        <line x1="55" y1="20" x2="55" y2="175" />
        {[0, 0.5, 1].map((t) => (
          <g key={t}>
            <line
              className="training-gridline"
              x1="55"
              x2="615"
              y1={175 - t * 150}
              y2={175 - t * 150}
            />
            <text x="47" y={179 - t * 150} textAnchor="end">
              {(t * ymax).toFixed(2)}
            </text>
          </g>
        ))}
        <polyline
          points={points
            .map(
              (r) => `${55 + (r[x] / xmax) * 560},${175 - (r[y] / ymax) * 150}`,
            )
            .join(" ")}
        />
        {points.map((r, i) => (
          <circle
            key={i}
            cx={55 + (r[x] / xmax) * 560}
            cy={175 - (r[y] / ymax) * 150}
            r="4"
          >
            <title>{`Step ${r[x]}: ${r[y].toFixed(5)}`}</title>
          </circle>
        ))}
        <text x="55" y="196">
          0
        </text>
        <text x="615" y="196" textAnchor="end">
          {xmax}
        </text>
        <text x="335" y="215" textAnchor="middle">
          Optimizer step
        </text>
      </svg>
      <figcaption>{label}</figcaption>
      <details>
        <summary>Exact curve values</summary>
        <div className="workflow-scroll">
          <table>
            <thead>
              <tr>
                <th>Step</th>
                <th>{label}</th>
              </tr>
            </thead>
            <tbody>
              {points.map((r, i) => (
                <tr key={i}>
                  <td>{r[x]}</td>
                  <td>{r[y].toFixed(6)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </figure>
  );
}
export function LossExplorer() {
  const [positive, setPositive] = useState(0.8),
    [negative, setNegative] = useState(0.4),
    [scale, setScale] = useState(5),
    [target, setTarget] = useState(1);
  const logits = [positive * scale, negative * scale],
    ce = crossEntropy(logits, 0),
    reg = cosineRegression([positive], [target]);
  return (
    <section
      className="reference-lab workflow-lab"
      aria-label="Embedding loss calculator"
    >
      <p className="workflow-kicker">
        Calculated teaching example · declared scores, no model run
      </p>
      <button
        onClick={() => {
          setPositive(0.8);
          setNegative(0.4);
          setScale(5);
          setTarget(1);
        }}
      >
        Reset experiment
      </button>
      <div className="reference-controls">
        <label>
          Positive similarity: {positive.toFixed(2)}
          <input
            aria-label="Positive pair similarity"
            type="range"
            min="-1"
            max="1"
            step=".05"
            value={positive}
            onChange={(e) => setPositive(+e.target.value)}
          />
        </label>
        <label>
          Alternative similarity: {negative.toFixed(2)}
          <input
            aria-label="Negative pair similarity"
            type="range"
            min="-1"
            max="1"
            step=".05"
            value={negative}
            onChange={(e) => setNegative(+e.target.value)}
          />
        </label>
        <label>
          Logit scale: {scale}
          <input
            aria-label="Training logit scale"
            type="range"
            min="1"
            max="40"
            value={scale}
            onChange={(e) => setScale(+e.target.value)}
          />
        </label>
        <label>
          Regression target
          <select
            aria-label="Cosine regression target"
            value={target}
            onChange={(e) => setTarget(+e.target.value)}
          >
            <option value="1">1 · same intent</option>
            <option value="0">0 · different intent</option>
          </select>
        </label>
      </div>
      <div className="workflow-stats" aria-live="polite">
        <Stat label="MNR one-anchor cross entropy" value={fixed(ce)} />
        <Stat label="Target candidate probability" value={pct(Math.exp(-ce))} />
        <Stat label="Cosine regression squared error" value={fixed(reg.loss)} />
      </div>
      <p>
        The MNR row targets candidate 0: scaled logits [
        {logits.map((x) => x.toFixed(2)).join(", ")}]. The cosine-regression
        target changes the desired value itself. Softmax pair classification
        instead learns a classifier over [u, v, |u−v|]; a two-dimensional
        illustration produces{" "}
        {JSON.stringify(
          pairFeatures([1, 0.5], [0.2, 0.8]).map((v) => Number(v.toFixed(2))),
        )}
        . These are different objectives, data contracts, and gradients.
      </p>
      <details>
        <summary>Actual library arithmetic checks from L15</summary>
        <Json value={data.embedding.loss_checks} />
      </details>
      <p>
        If the alternative is another valid answer, MNR still penalizes it under
        a single-positive target. A clean diagonal label does not make an
        in-batch negative semantically correct.
      </p>
    </section>
  );
}
export function EmbeddingTraining() {
  const d = data.embedding,
    [kind, setKind] = useState("mnr"),
    [point, setPoint] = useState(0),
    r = d.conditions.find((r: any) => r.kind === kind),
    p = r.projection[point];
  const coords = d.conditions.flatMap((condition: any) =>
      condition.projection.flatMap((r: any) => [r.before, r.after]),
    ),
    xmin = Math.min(...coords.map((p: number[]) => p[0])),
    xmax = Math.max(...coords.map((p: number[]) => p[0])),
    ymin = Math.min(...coords.map((p: number[]) => p[1])),
    ymax = Math.max(...coords.map((p: number[]) => p[1]));
  const x = (v: number) => 40 + ((v - xmin) / (xmax - xmin || 1)) * 500,
    y = (v: number) => 230 - ((v - ymin) / (ymax - ymin || 1)) * 200;
  return (
    <section
      className="reference-lab workflow-lab"
      aria-label="Actual embedding training results"
    >
      <p className="workflow-kicker">
        L15 · actual CPU optimization · saved and reloaded encoders
      </p>
      <button
        onClick={() => {
          setKind("mnr");
          setPoint(0);
        }}
      >
        Reset experiment
      </button>
      <div className="reference-controls">
        <label>
          Objective
          <select
            aria-label="Embedding training objective"
            value={kind}
            onChange={(e) => setKind(e.target.value)}
          >
            <option value="softmax">Pair softmax classifier</option>
            <option value="cosine">Cosine similarity regression</option>
            <option value="mnr">Multiple negatives ranking</option>
          </select>
        </label>
        <label>
          Inspect a test sentence
          <select
            aria-label="Embedding projection sentence"
            value={point}
            onChange={(e) => setPoint(+e.target.value)}
          >
            {r.projection.map((p: any, i: number) => (
              <option key={p.id} value={i}>
                {p.intent} · {p.id}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="workflow-stats" aria-live="polite">
        <Stat
          label="Baseline test retrieval MRR"
          value={fixed(d.baseline.retrieval.mrr)}
        />
        <Stat label="Selected model test MRR" value={fixed(r.retrieval.mrr)} />
        <Stat label="Selected test pair MSE" value={fixed(r.similarity.mse)} />
        <Stat
          label="Selected optimizer step"
          value={r.training.selected_step}
        />
      </div>
      <p>
        {r.training.selection}. Validation chooses checkpoints; test predictions
        are inspected after selection. Original same-intent pairs are not an NLI
        or general STS benchmark.
      </p>
      <Curve
        rows={r.training.history}
        y="training_loss"
        label="Observed training batch loss"
      />
      <Curve
        rows={r.training.history}
        y="validation_mrr"
        label="Validation retrieval MRR"
        unit="quality"
      />
      <figure className="training-scatter">
        <svg
          viewBox="0 0 580 270"
          role="img"
          aria-label="Fixed PCA projection of actual before and after embeddings"
        >
          <title>
            Projected actual unit embeddings; outlined circles before, filled
            circles after; selected sentence has a connecting line.
          </title>
          {r.projection.map((q: any, i: number) => (
            <g key={q.id}>
              <circle
                cx={x(q.before[0])}
                cy={y(q.before[1])}
                r="5"
                className="training-before"
              />
              <circle
                cx={x(q.after[0])}
                cy={y(q.after[1])}
                r={i === point ? 8 : 4}
              />
            </g>
          ))}
          <line
            x1={x(p.before[0])}
            y1={y(p.before[1])}
            x2={x(p.after[0])}
            y2={y(p.after[1])}
          />
          <text x="290" y="262" textAnchor="middle">
            Frozen baseline-training PCA · component 1
          </text>
        </svg>
        <figcaption>
          {p.text} · {p.intent}. Outline: before; filled: after. These two
          dimensions omit information.
        </figcaption>
      </figure>
      <details>
        <summary>
          Gradients, changed parameters, and full-dimensional evaluation
        </summary>
        <Json
          value={{
            parameters: r.training.parameters,
            similarity: r.similarity,
            retrieval: r.retrieval,
            projection_policy: d.projection_policy,
          }}
        />
      </details>
    </section>
  );
}
export function SilverExplorer() {
  const d = data.augmentation,
    [index, setIndex] = useState(0),
    s = d.silver[index],
    diagnostic = d.diagnostics[index];
  return (
    <section
      className="reference-lab workflow-lab"
      aria-label="Teacher and silver label explorer"
    >
      <p className="workflow-kicker">
        L16 · trained cross-encoder → actual silver scores → two trained
        students
      </p>
      <button onClick={() => setIndex(0)}>Reset experiment</button>
      <label>
        Training-pool candidate
        <select
          aria-label="Silver label candidate"
          value={index}
          onChange={(e) => setIndex(+e.target.value)}
        >
          {d.silver.map((r: any, i: number) => (
            <option value={i} key={r.id}>
              {r.id}
              {d.diagnostics[i].disagreement ? " · teacher disagreement" : ""}
            </option>
          ))}
        </select>
      </label>
      <div className="training-pair">
        <blockquote>
          <strong>{s.left_id}</strong>
          <p>{s.left}</p>
        </blockquote>
        <blockquote>
          <strong>{s.right_id}</strong>
          <p>{s.right}</p>
        </blockquote>
      </div>
      <div className="workflow-stats" aria-live="polite">
        <Stat
          label="Teacher probability supplied to student"
          value={fixed(s.label)}
        />
        <Stat
          label="Latent author relation · diagnostic only"
          value={
            diagnostic.latent_same_intent ? "Same intent" : "Different intents"
          }
        />
        <Stat
          label="Threshold 0.5 diagnostic"
          value={diagnostic.disagreement ? "Disagrees" : "Agrees"}
        />
      </div>
      <p>
        The student trains on the continuous teacher score, not this thresholded
        diagnostic. Candidate sampling and teacher inference never use latent
        intent labels or evaluation sentences. A confident silver score can
        still be wrong.
      </p>
      <div className="workflow-scroll">
        <table>
          <thead>
            <tr>
              <th>Student</th>
              <th>Training pairs</th>
              <th>Test MSE ↓</th>
              <th>Test retrieval MRR ↑</th>
            </tr>
          </thead>
          <tbody>
            {d.students.map((r: any) => (
              <tr key={r.condition}>
                <td>{r.condition}</td>
                <td>{r.train_pairs}</td>
                <td>{fixed(r.similarity.mse)}</td>
                <td>{fixed(r.retrieval.mrr)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p>{d.comparison_limit}</p>
      <Curve
        rows={d.teacher.history}
        y="validation_binary_ce"
        label="Teacher validation binary cross entropy"
      />
      <details>
        <summary>
          Teacher selection, raw silver provenance and diagnostics
        </summary>
        <Json
          value={{
            teacher: d.teacher,
            selected_candidate: s,
            diagnostic,
            total_teacher_disagreements: d.diagnostics.filter(
              (r: any) => r.disagreement,
            ).length,
          }}
        />
      </details>
    </section>
  );
}
export function DenoisingExplorer() {
  const d = data.denoising,
    [index, setIndex] = useState(0),
    r = d.corruption[index];
  return (
    <section
      className="reference-lab workflow-lab"
      aria-label="Actual denoising and adaptation"
    >
      <p className="workflow-kicker">
        L17 · actual tied TSDAE · reconstruction is a surrogate
      </p>
      <button onClick={() => setIndex(0)}>Reset experiment</button>
      <label>
        Recorded corrupted training sentence
        <select
          aria-label="Denoising sentence"
          value={index}
          onChange={(e) => setIndex(+e.target.value)}
        >
          {d.corruption.map((r: any, i: number) => (
            <option key={r.id} value={i}>
              {r.id}
            </option>
          ))}
        </select>
      </label>
      <div className="training-pair">
        <blockquote>
          <strong>Original target</strong>
          <p>{r.original}</p>
        </blockquote>
        <blockquote>
          <strong>Actual damaged input · seed {r.seed}</strong>
          <p>{r.damaged}</p>
        </blockquote>
      </div>
      <p>
        Retained whitespace-word indices: {r.retained_word_indices.join(", ")}.
        Nominal deletion probability {pct(r.deletion_probability)}; at least one
        word remains. The single CLS vector conditions a causal decoder.
        Training uses gold previous tokens; greedy reconstruction does not.
      </p>
      <Curve
        rows={d.history}
        y="reconstruction_training_probe_loss"
        label="Fixed corruption training reconstruction loss"
      />
      <Curve
        rows={d.history}
        y="validation_mrr"
        label="Separate validation retrieval MRR"
        unit="quality"
      />
      <div className="workflow-stats">
        <Stat
          label="Baseline test retrieval MRR"
          value={fixed(d.baseline.retrieval.mrr)}
        />
        <Stat
          label="TSDAE-selected test MRR"
          value={fixed(d.tsdae.retrieval.mrr)}
        />
        <Stat
          label="Loss decreases without MRR improvement"
          value={`${d.counterexamples.length} intervals`}
        />
      </div>
      <details>
        <summary>
          Actual greedy reconstruction · fixed first training probe
        </summary>
        <p>
          This saved generation belongs to “{d.reconstruction.source.original}”,
          with damaged input “{d.reconstruction.source.damaged}”. Changing the
          selection above does not rerun its decoder.
        </p>
        <pre className="workflow-pre">{d.reconstruction.greedy}</pre>
        <Json value={d.reconstruction} />
      </details>
      <div className="workflow-scroll">
        <table>
          <thead>
            <tr>
              <th>Subsequent supervised adaptation</th>
              <th>Test MSE ↓</th>
              <th>Test retrieval MRR ↑</th>
            </tr>
          </thead>
          <tbody>
            {d.adaptation.map((r: any) => (
              <tr key={r.condition}>
                <td>{r.condition}</td>
                <td>{fixed(r.similarity.mse)}</td>
                <td>{fixed(r.retrieval.mrr)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p>
        Gold-only and TSDAE-plus-gold use the same supervised budget and CLS
        pooling. Their trained parameters, checkpoints and artifact reloads are
        independently checked. No general improvement claim follows from this
        small original corpus.
      </p>
    </section>
  );
}
export function FreezeExplorer() {
  const d = data.classification,
    [seed, setSeed] = useState(11),
    [method, setMethod] = useState("bert-partial"),
    [index, setIndex] = useState(0),
    r = d.runs.find((r: any) => r.seed === seed && r.method === method),
    p = r.predictions[index];
  return (
    <section
      className="reference-lab workflow-lab"
      aria-label="Freezing and few-shot classification"
    >
      <p className="workflow-kicker">
        L18 · three fixed selections · real BERT and SetFit updates
      </p>
      <button
        onClick={() => {
          setSeed(11);
          setMethod("bert-partial");
          setIndex(0);
        }}
      >
        Reset experiment
      </button>
      <div className="reference-controls">
        <label>
          Few-shot selection seed
          <select
            aria-label="Few-shot seed"
            value={seed}
            onChange={(e) => setSeed(+e.target.value)}
          >
            {[11, 23, 37].map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
        </label>
        <label>
          Method
          <select
            aria-label="Representation classifier"
            value={method}
            onChange={(e) => setMethod(e.target.value)}
          >
            {d.summary.map((r: any) => (
              <option key={r.method}>{r.method}</option>
            ))}
          </select>
        </label>
      </div>
      <div className="workflow-stats" aria-live="polite">
        <Stat label="Training labels" value="8 per class" />
        <Stat
          label="Test macro-F1 · selected seed"
          value={pct(r.test.macro_f1)}
        />
        <Stat label="Test accuracy" value={pct(r.test.accuracy)} />
      </div>
      {r.parameters && (
        <div className="training-layers">
          {Object.entries(r.parameters).map(([name, v]: [string, any]) => (
            <div
              key={name}
              className={v.trainable ? "training-updated" : "training-frozen"}
            >
              <strong>{name.replaceAll("_", " ")}</strong>
              <span>
                {v.trainable.toLocaleString()} / {v.parameters.toLocaleString()}{" "}
                trainable parameters
              </span>
              <span>
                {v.changed_tensors} changed tensors · first gradient norm{" "}
                {v.gradient_norm.toFixed(5)}
              </span>
            </div>
          ))}
        </div>
      )}
      {r.history && (
        <Curve
          rows={r.history}
          y={method === "setfit" ? "embedding_loss" : "validation_macro_f1"}
          label={
            method === "setfit"
              ? "SetFit observed embedding loss"
              : "Validation classification macro-F1"
          }
          unit={method === "setfit" ? "loss" : "quality"}
        />
      )}
      <label>
        Actual test prediction
        <select
          aria-label="Classification prediction"
          value={index}
          onChange={(e) => setIndex(+e.target.value)}
        >
          {r.predictions.map((p: any, i: number) => (
            <option key={p.id} value={i}>
              {p.id} · {p.prediction === p.gold ? "correct" : "error"}
            </option>
          ))}
        </select>
      </label>
      <blockquote>
        {p.text ??
          "Original review text is available by its stable ID in the shared sentiment fixture."}
      </blockquote>
      <p>
        <strong>{p.id}</strong> · gold {p.gold === 1 ? "positive" : "negative"};
        predicted {p.prediction === 1 ? "positive" : "negative"}.
      </p>
      <details>
        <summary>Per-class scores, selection IDs and all-seed summary</summary>
        <Json
          value={{
            test: r.test,
            train_ids: r.train_ids,
            summary: d.summary,
            protocol: d.protocol,
            settings: r.settings,
            earlier_reference: d.earlier_full_data_tfidf,
          }}
        />
      </details>
      <p>
        More generated pairs do not create more independent labeled reviews.
        Differences among seeds expose sampling variability. MiniLM sentence
        encoders and tiny BERT differ in architecture and prior training, so the
        entire comparison does not isolate one objective.
      </p>
    </section>
  );
}
export function MaskExplorer() {
  const d = data.mlm,
    original = d.mask_trace,
    [selected, setSelected] = useState<number[]>(
      original.flatMap((r: any, i: number) => (r.loss_included ? [i] : [])),
    ),
    labels = supervisedMask(original, selected);
  const toggle = (i: number) =>
    setSelected(
      selected.includes(i) ? selected.filter((v) => v !== i) : [...selected, i],
    );
  return (
    <section
      className="reference-lab workflow-lab"
      aria-label="Masked language model supervision"
    >
      <p className="workflow-kicker">
        L19 · actual masked batch; editable mask is a calculated exercise
      </p>
      <button
        onClick={() =>
          setSelected(
            original.flatMap((r: any, i: number) =>
              r.loss_included ? [i] : [],
            ),
          )
        }
      >
        Reset experiment
      </button>
      <p>
        Select eligible token positions to change the illustrated loss mask.
        This does not alter the recorded corruption or rerun training.
        Special/padding tokens stay excluded. Original selected-but-unchanged
        tokens still receive a target.
      </p>
      <div className="training-tokens">
        {original.map((r: any, i: number) => (
          <button
            key={i}
            disabled={r.special_or_padding}
            aria-pressed={selected.includes(i)}
            aria-label={`Supervise position ${i}: ${r.original_token}`}
            onClick={() => toggle(i)}
          >
            <span>{r.input_token}</span>
            <small>
              {labels[i] === -100 ? "ignored" : `target ${r.original_token}`}
            </small>
          </button>
        ))}
      </div>
      <div className="workflow-stats" aria-live="polite">
        <Stat
          label="Illustrated supervised positions"
          value={selected.length}
        />
        <Stat
          label="Original run selected positions"
          value={original.filter((r: any) => r.loss_included).length}
        />
      </div>
      {!selected.length && (
        <p role="status">
          No target tokens remain. A mean loss would be undefined; the actual
          trainer requires at least one target in each batch.
        </p>
      )}
      <p>{d.masking}</p>
      <Curve
        rows={d.evaluated.history}
        y="validation_masked_loss"
        label="Fixed masked validation token loss"
      />
      <div className="workflow-scroll">
        <table>
          <thead>
            <tr>
              <th>Downstream starting encoder</th>
              <th>Test macro-F1</th>
              <th>Equal head/pooler initialization</th>
            </tr>
          </thead>
          <tbody>
            {d.downstream.map((r: any) => (
              <tr key={r.condition}>
                <td>{r.condition}</td>
                <td>{pct(r.test.macro_f1)}</td>
                <td>
                  {r.head_pooler_initial_sha256 ===
                  d.downstream[0].head_pooler_initial_sha256
                    ? "Verified"
                    : "Different"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <details>
        <summary>
          Actual mask IDs, smoke/evaluated profiles and source split IDs
        </summary>
        <Json
          value={{
            actual: original,
            illustrated_labels: labels,
            smoke: d.smoke,
            evaluated: d.evaluated,
            training_ids: d.training_ids,
            validation_ids: d.validation_ids,
            test_ids: d.test_ids,
          }}
        />
      </details>
      <p>
        The two-step smoke run is not the evaluated adaptation. Test reviews
        never enter MLM updates. Fill-mask output and reconstruction loss are
        not substitutes for downstream classification checks.
      </p>
    </section>
  );
}
export function NERAlignment() {
  const d = data.ner,
    [probe, setProbe] = useState("split-and-multiword"),
    [cap, setCap] = useState(32),
    [policy, setPolicy] = useState("first"),
    [index, setIndex] = useState(0),
    r = d.alignment.find((r: any) => r.id === probe && r.max_length === cap),
    prediction = d.training.predictions[index];
  return (
    <section
      className="reference-lab workflow-lab"
      aria-label="Word subword and entity alignment"
    >
      <p className="workflow-kicker">
        L20 · actual tokenizer mappings · reused actual NER trainer
      </p>
      <button
        onClick={() => {
          setProbe("split-and-multiword");
          setCap(32);
          setPolicy("first");
          setIndex(0);
        }}
      >
        Reset experiment
      </button>
      <div className="reference-controls">
        <label>
          Annotation probe
          <select
            aria-label="NER alignment input"
            value={probe}
            onChange={(e) => setProbe(e.target.value)}
          >
            <option value="split-and-multiword">
              Split names + multiword entities
            </option>
            <option value="no-entities">No entities</option>
          </select>
        </label>
        <label>
          Input cap
          <select
            aria-label="NER token limit"
            value={cap}
            onChange={(e) => setCap(+e.target.value)}
          >
            <option value="32">32 · padded complete input</option>
            <option value="6">6 · truncation probe</option>
          </select>
        </label>
        <label>
          Supervision policy
          <select
            aria-label="NER subtoken supervision"
            value={policy}
            onChange={(e) => setPolicy(e.target.value)}
          >
            <option value="first">
              First complete subtoken only · trained route
            </option>
            <option value="all">Propagate B→I · alignment alternative</option>
          </select>
        </label>
      </div>
      <p>{r.tokens.join(" ")}</p>
      {r.training_rejected && (
        <p className="training-rejection" role="status">
          Training rejected: truncation removes a word or part of an annotated
          entity. Chunk at safe entity boundaries before training. The
          first-subtoken inspection mask below is not approval to train this
          shortened record.
        </p>
      )}
      <div className="workflow-scroll">
        <table>
          <thead>
            <tr>
              <th>Word ID</th>
              <th>Word / gold BIO</th>
              <th>Actual piece / ID</th>
              <th>Training label</th>
              <th>Loss</th>
            </tr>
          </thead>
          <tbody>
            {r.rows.map((p: any) => (
              <tr
                key={p.position}
                className={
                  policy === "first"
                    ? p.loss_first
                      ? "training-supervised"
                      : ""
                    : p.loss_all
                      ? "training-supervised"
                      : ""
                }
              >
                <td>{p.word_id ?? "special/pad"}</td>
                <td>
                  {p.word ?? "—"} {p.word_tag ?? ""}
                </td>
                <td>
                  {p.piece} · {p.token_id}
                </td>
                <td>
                  {policy === "first"
                    ? p.label_first
                    : (p.label_all ?? "rejected")}
                </td>
                <td>
                  {policy === "first"
                    ? p.loss_first
                      ? "included"
                      : "ignored"
                    : p.loss_all === null
                      ? "rejected"
                      : p.loss_all
                        ? "included"
                        : "ignored"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p>
        Labels use the fixed O/PER/ORG/LOC BIO vocabulary. −100 ignores special
        tokens, padding, and later subpieces under first-only supervision.
        Propagation changes repeated B tags to I within the same word. Only
        first-only supervision was used in this saved trained model.
      </p>
      <label>
        Actual held-out sentence
        <select
          aria-label="NER held-out prediction"
          value={index}
          onChange={(e) => setIndex(+e.target.value)}
        >
          {d.training.predictions.map((r: any, i: number) => (
            <option key={r.id} value={i}>
              {r.id}
            </option>
          ))}
        </select>
      </label>
      <p>{prediction.tokens.join(" ")}</p>
      <Json
        value={{
          gold: prediction.gold,
          predicted: prediction.predicted,
          gold_spans: prediction.gold_spans,
          predicted_spans: prediction.predicted_spans,
        }}
      />
      <div className="workflow-stats">
        <Stat
          label="Strict actual entity F1"
          value={pct(d.training.metrics.f1)}
        />
        <Stat
          label="All-O baseline entity F1"
          value={pct(d.training.all_o_baseline.f1)}
        />
        <Stat
          label="Actual token accuracy"
          value={pct(d.training.token_accuracy)}
        />
      </div>
      <details>
        <summary>Boundary, type and invalid-BIO teaching cases</summary>
        <Json value={d.span_examples} />
      </details>
      <p>{d.scope}</p>
    </section>
  );
}
