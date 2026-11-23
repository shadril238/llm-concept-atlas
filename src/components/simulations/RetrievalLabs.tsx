import { useState } from "react";
import published from "../../data/retrieval-results.json";
import {
  rankingMetrics,
  makeChunks,
  candidateSoftmax,
  contrastiveLoss,
  citationMetrics,
} from "../../lib/retrieval-mechanics.js";
const data: Record<string, any> = published;
const percent = (n: number | null) =>
  n === null ? "undefined" : (100 * n).toFixed(1) + "%";
function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="workflow-stat">
      <strong>{value}</strong>
      <span>{label}</span>
    </div>
  );
}
function Json({ value }: { value: unknown }) {
  return <pre className="workflow-pre">{JSON.stringify(value, null, 2)}</pre>;
}
function Call({
  call,
  label = "Actual model call",
}: {
  call: any;
  label?: string;
}) {
  return (
    <details>
      <summary>
        {label} · {call.prompt_tokens ?? call.input_tokens} input tokens
      </summary>
      <p>Unedited output</p>
      <pre className="workflow-pre">{call.raw}</pre>
      <p>Actual prompt</p>
      <pre className="workflow-pre">{call.prompt ?? call.rendered_prompt}</pre>
      <Json
        value={
          call.settings ?? {
            choices: call.choices,
            mean_log_probabilities: call.mean_log_probabilities,
            scored_tokens: call.scored_tokens,
          }
        }
      />
    </details>
  );
}
const base = "/llm-concept-atlas";
export function RetrievalExplorer() {
  const d = data.retrieval,
    [queryId, setQuery] = useState("eval-hours"),
    [method, setMethod] = useState("dense"),
    [k, setK] = useState(3);
  const query = d.queries.find((q: any) => q.id === queryId),
    row = d.rows.find((r: any) => r.id === queryId && r.method === method),
    metrics = rankingMetrics(row.document_ranking, query.relevance, k);
  const seen = new Set<string>();
  const hits = row.hits
    .filter((r: any) => {
      if (seen.has(r.document_id)) return false;
      seen.add(r.document_id);
      return true;
    })
    .slice(0, k);
  return (
    <section
      className="reference-lab workflow-lab"
      aria-label="Verified retrieval benchmark"
    >
      <p className="workflow-kicker">
        L12 · actual frozen encoder and cross-encoder · recorded rankings
      </p>
      <button
        onClick={() => {
          setQuery("eval-hours");
          setMethod("dense");
          setK(3);
        }}
      >
        Reset experiment
      </button>
      <div className="reference-controls">
        <label>
          Query
          <select
            aria-label="Retrieval query"
            value={queryId}
            onChange={(e) => setQuery(e.target.value)}
          >
            {d.queries.map((q: any) => (
              <option key={q.id} value={q.id}>
                {q.id} · {q.kind}
              </option>
            ))}
          </select>
        </label>
        <label>
          Method
          <select
            aria-label="Retrieval method"
            value={method}
            onChange={(e) => setMethod(e.target.value)}
          >
            <option value="lexical">BM25</option>
            <option value="dense">Dense cosine index</option>
            <option value="reranked">
              Dense top-six chunks → cross-encoder
            </option>
          </select>
        </label>
        <label>
          Cutoff k: {k}
          <input
            aria-label="Retrieval cutoff"
            type="range"
            min="1"
            max="5"
            value={k}
            onChange={(e) => setK(Number(e.target.value))}
          />
        </label>
      </div>
      <p>
        <strong>{query.text}</strong> · {query.split}.{" "}
        {query.answerable
          ? "The corpus contains relevant evidence."
          : "No answer evidence exists in this corpus."}
      </p>
      <div className="workflow-stats" aria-live="polite">
        <Stat label={`Precision@${k}`} value={percent(metrics.precision)} />
        <Stat label={`Recall@${k}`} value={percent(metrics.recall)} />
        <Stat label={`AP@${k}`} value={percent(metrics.ap)} />
        <Stat label={`nDCG@${k}`} value={percent(metrics.ndcg)} />
      </div>
      <p>
        Metrics use <strong>unique document IDs</strong>, not repeated
        overlapping chunks. AP divides by all {metrics.relevant_documents}{" "}
        relevant documents. Undefined absent-query AP is excluded from MAP, with
        its denominator reported.
      </p>
      <ol className="retrieval-hits">
        {hits.map((hit: any) => (
          <li key={hit.id}>
            <div className="retrieval-hit-heading">
              <strong>{hit.title}</strong>
              <span className="retrieval-grade">
                grade {query.relevance[hit.document_id] ?? 0}
              </span>
            </div>
            <p>{hit.text}</p>
            <small>
              {hit.id} · source offsets [{hit.start},{hit.end}) · score{" "}
              {hit.score.toFixed(4)}
            </small>
            <details>
              <summary>Inspect original source and candidate values</summary>
              <p>
                {d.documents.find((s: any) => s.id === hit.document_id).text}
              </p>
              <Json
                value={{
                  document_id: hit.document_id,
                  source_uri: hit.source_uri,
                  dense_cosine: hit.dense_score,
                  ranking_score: hit.score,
                }}
              />
            </details>
          </li>
        ))}
      </ol>
      <details>
        <summary>Compare held-out methods and the design threshold</summary>
        <div className="workflow-scroll">
          <table>
            <thead>
              <tr>
                <th>Method</th>
                <th>k</th>
                <th>MAP</th>
                <th>nDCG</th>
                <th>AP queries</th>
              </tr>
            </thead>
            <tbody>
              {d.aggregates.map((r: any, i: number) => (
                <tr key={i}>
                  <td>{r.method}</td>
                  <td>{r.k}</td>
                  <td>{percent(r.ap)}</td>
                  <td>{percent(r.ndcg)}</td>
                  <td>
                    {r.answerable_queries} / {r.queries}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <Json value={d.threshold_selection} />
        <p>
          BM25, cosine, and raw pair logits have different scales. Reranking can
          only reorder its six input chunks. Related grade-one documents are
          useful to retrieval evaluation but may not contain enough evidence for
          a full answer.
        </p>
      </details>
    </section>
  );
}
export function ChunkExplorer() {
  const docs = data.retrieval.documents,
    [id, setId] = useState("loans"),
    [mode, setMode] = useState("sentence"),
    [size, setSize] = useState(2),
    [overlap, setOverlap] = useState(1),
    [selected, setSelected] = useState(0);
  const doc = docs.find((d: any) => d.id === id),
    parts = makeChunks(doc.text, size, overlap, mode),
    part = parts[Math.min(selected, parts.length - 1)];
  return (
    <section
      className="reference-lab workflow-lab"
      aria-label="Source-preserving chunk calculator"
    >
      <p className="workflow-kicker">
        Calculated boundaries · no encoder run in your browser
      </p>
      <button
        onClick={() => {
          setId("loans");
          setMode("sentence");
          setSize(2);
          setOverlap(1);
          setSelected(0);
        }}
      >
        Reset experiment
      </button>
      <div className="reference-controls">
        <label>
          Source
          <select
            aria-label="Chunk source"
            value={id}
            onChange={(e) => {
              setId(e.target.value);
              setSelected(0);
            }}
          >
            {docs.map((d: any) => (
              <option value={d.id} key={d.id}>
                {d.title}
              </option>
            ))}
          </select>
        </label>
        <label>
          Unit
          <select
            aria-label="Chunk unit"
            value={mode}
            onChange={(e) => {
              const word = e.target.value === "word";
              setMode(e.target.value);
              setSize(word ? 12 : 2);
              setOverlap(word ? 3 : 1);
              setSelected(0);
            }}
          >
            <option value="sentence">Sentence</option>
            <option value="word">Whitespace word</option>
          </select>
        </label>
        <label>
          Units per chunk: {size}
          <input
            aria-label="Chunk size"
            type="range"
            min="1"
            max={mode === "word" ? 24 : 3}
            value={size}
            onChange={(e) => {
              const n = Number(e.target.value);
              setSize(n);
              setOverlap(Math.min(overlap, n - 1));
              setSelected(0);
            }}
          />
        </label>
        <label>
          Overlap: {overlap}
          <input
            aria-label="Chunk overlap"
            type="range"
            min="0"
            max={size - 1}
            value={overlap}
            onChange={(e) => {
              setOverlap(Number(e.target.value));
              setSelected(0);
            }}
          />
        </label>
      </div>
      <p className="retrieval-source">
        {Array.from(doc.text as string).slice(0, part.start).join("")}
        <mark className="retrieval-highlight">{part.text}</mark>
        {Array.from(doc.text as string).slice(part.end).join("")}
      </p>
      <div className="workflow-path">
        {parts.map((c, i) => (
          <button
            key={i}
            aria-pressed={i === Math.min(selected, parts.length - 1)}
            onClick={() => setSelected(i)}
          >
            Chunk {i + 1}: [{c.start},{c.end})
          </button>
        ))}
      </div>
      <Json value={part} />
      <p>
        Offsets are zero-based Unicode code points, end exclusive. Sentence
        boundaries use a documented punctuation/whitespace rule; they are not a
        general linguistic parser. Word counts differ from model token counts.
        The verified L12 pipeline separately checks its native tokenizer cap and
        compares two-sentence versus 24-word windows.
      </p>
    </section>
  );
}
export function MetricExplorer() {
  const [order, setOrder] = useState(["A", "B", "C", "D"]),
    [k, setK] = useState(3);
  const grades: Record<string, number> = { A: 2, B: 0, C: 1, D: 2 },
    m = rankingMetrics(order, grades, k);
  function move(index: number, direction: number) {
    const next = [...order],
      target = index + direction;
    if (target < 0 || target >= next.length) return;
    [next[index], next[target]] = [next[target], next[index]];
    setOrder(next);
  }
  return (
    <section
      className="reference-lab workflow-lab"
      aria-label="Ranking metric calculator"
    >
      <p className="workflow-kicker">
        Calculated teaching example · fixed relevance judgments
      </p>
      <button
        onClick={() => {
          setOrder(["A", "B", "C", "D"]);
          setK(3);
        }}
      >
        Reset experiment
      </button>
      <label>
        Cutoff k: {k}
        <input
          aria-label="Metric cutoff"
          type="range"
          min="1"
          max="4"
          value={k}
          onChange={(e) => setK(Number(e.target.value))}
        />
      </label>
      <div className="workflow-stats" aria-live="polite">
        <Stat label="P@k" value={percent(m.precision)} />
        <Stat label="AP@k" value={percent(m.ap)} />
        <Stat label="nDCG@k" value={percent(m.ndcg)} />
      </div>
      <ol>
        {order.map((id, i) => (
          <li key={id}>
            Document {id} · grade {grades[id]}{" "}
            <button
              disabled={i === 0}
              aria-label={`Move ${id} up`}
              onClick={() => move(i, -1)}
            >
              Move up
            </button>{" "}
            <button
              disabled={i === order.length - 1}
              aria-label={`Move ${id} down`}
              onClick={() => move(i, 1)}
            >
              Move down
            </button>
            {i >= k ? " · beyond cutoff" : ""}
          </li>
        ))}
      </ol>
      <p>
        AP contributions: {m.ap_terms.map((v) => v.toFixed(3)).join(" + ")};
        divide their sum by <strong>{m.relevant_documents}</strong>, the total
        relevant documents, including D if it falls beyond k. DCG uses gain
        2^grade − 1 and log2(rank + 1). nDCG divides by the best possible
        ordering under the same cutoff.
      </p>
      <p>
        MAP averages these per-query AP values across queries with defined AP.
        MRR only uses the first relevant rank and therefore cannot summarize how
        well later relevant documents are ordered.
      </p>
    </section>
  );
}
export function GroundedExplorer() {
  const d = data.grounded,
    [id, setId] = useState("eval-hours"),
    [variant, setVariant] = useState("baseline");
  const ids = [...new Set(d.rows.map((r: any) => r.id))] as string[],
    r = d.rows.find((r: any) => r.id === id && r.variant === variant),
    review = d.annotations.rows.find(
      (v: any) => v.id === id && v.variant === variant,
    );
  return (
    <section
      className="reference-lab workflow-lab"
      aria-label="Actual grounded retrieval trace"
    >
      <p className="workflow-kicker">
        L13 · actual local calls · author-known mechanism probes
      </p>
      <button
        onClick={() => {
          setId("eval-hours");
          setVariant("baseline");
        }}
      >
        Reset experiment
      </button>
      <div className="reference-controls">
        <label>
          Question
          <select
            aria-label="Grounded question"
            value={id}
            onChange={(e) => setId(e.target.value)}
          >
            {ids.map((id) => (
              <option key={id}>{id}</option>
            ))}
          </select>
        </label>
        <label>
          Workflow
          <select
            aria-label="Grounded workflow"
            value={variant}
            onChange={(e) => setVariant(e.target.value)}
          >
            {[
              "baseline",
              "rewrite",
              "multi-query",
              "multi-hop",
              "routing",
              "agentic",
            ].map((v) => (
              <option key={v}>{v}</option>
            ))}
          </select>
        </label>
      </div>
      <p>
        <strong>{r.question}</strong>
      </p>
      <div className="workflow-stats" aria-live="polite">
        <Stat label="Model decision/generation calls" value={r.model_calls} />
        <Stat label="Actual searches" value={r.search_calls} />
        <Stat label="Generated tokens" value={r.output_tokens} />
        <Stat
          label="Finite-label scoring tokens"
          value={r.scored_tokens ?? 0}
        />
      </div>
      <ol className="retrieval-stages">
        <li>
          <strong>Plan and search · {r.plan.stop_reason}</strong>
          <p>
            Rewrites and follow-up queries below are actual generated strings.
            Multi-hop uses earlier retrieved evidence. Agentic mode scores a
            finite source domain, then executes searches with the original
            question.
          </p>
          {r.plan.searches.map((s: any, i: number) => (
            <details key={i}>
              <summary>
                Search {i + 1}: {s.query} · {s.collection}
              </summary>
              <ol>
                {s.hits.slice(0, 3).map((h: any) => (
                  <li key={h.id}>
                    {h.id} · {h.dense_score.toFixed(3)}
                    <p>{h.text}</p>
                  </li>
                ))}
              </ol>
            </details>
          ))}
          <details>
            <summary>Planner decisions and actual model inputs</summary>
            <Json value={r.plan.trace} />
            {r.plan.calls.map((c: any, i: number) => (
              <Call key={i} call={c} label={`Planner ${i + 1}`} />
            ))}
          </details>
        </li>
        <li>
          <strong>
            Pack context · {r.answer_call.prompt_tokens} input tokens
          </strong>
          <p>
            Whole passages fit a {d.context_budget}-token application budget.
            Duplicate documents are removed. Design-only cosine threshold:{" "}
            {d.threshold}. It is not a calibrated probability.
          </p>
          {r.context.length ? (
            r.context.map((s: any) => (
              <blockquote key={s.id}>
                <strong>[{s.id}]</strong> {s.text}
              </blockquote>
            ))
          ) : (
            <p>
              No evidence supplied. The threshold/controller withheld context.
            </p>
          )}
          <details>
            <summary>Omitted passages and reasons</summary>
            <Json value={r.omitted} />
          </details>
        </li>
        <li>
          <strong>Actual raw draft</strong>
          <pre className="workflow-pre">{r.answer_call.raw}</pre>
          <p>
            Model citation IDs: {r.model_citations.join(", ") || "none"}.{" "}
            {r.citation_count
              ? `All resolve: ${r.citation_ids_resolve}; all from context: ${r.citations_from_selected_context}.`
              : "No citation-support claim is established."}
          </p>
          <Call call={r.answer_call} />
        </li>
        <li>
          <strong>Application display policy</strong>
          <pre className="workflow-pre">{r.display_answer}</pre>
          <p>
            Controller withheld: {String(r.controller_withheld)}. Model said
            UNKNOWN: {String(r.model_abstained)}. A withheld hallucination does
            not become a grounded model answer.
          </p>
        </li>
        <li>
          <strong>Reference and author proposition review</strong>
          <Json
            value={{
              reference: r.reference_answer,
              required_support_documents: r.reference_support_ids,
              support_document_recall: r.support_document_recall,
            }}
          />
          {review ? (
            <>
              <p>{review.finding}</p>
              <Json
                value={{
                  answer_correct: review.answer_correct,
                  context_faithfulness: review.context_faithfulness,
                  fluency: review.fluency,
                  utility: review.utility,
                  citation_metrics: review.metrics,
                }}
              />
              <p>
                Author judgment of the raw draft; independent review is pending.
                Unknown entailment remains null.
              </p>
            </>
          ) : (
            <p>No proposition review is supplied for this case yet.</p>
          )}
        </li>
      </ol>
    </section>
  );
}
export function CitationAuditor() {
  const [chosen, setChosen] = useState<string[][]>([
    ["hours"],
    ["loans"],
    ["birch"],
  ]);
  const claims = [
    { text: "The library closes at 17:30 on Saturday.", needed: ["hours"] },
    { text: "Wren seats twelve people.", needed: ["wren"] },
    {
      text: "The team maintaining Wren can be reached at extension 42.",
      needed: ["wren", "birch"],
    },
  ];
  const judged = claims.map((c, i) => ({
      fully_supported_by_citations: c.needed.every((id) =>
        chosen[i].includes(id),
      ),
      citations: chosen[i].map((id) => ({ supports: c.needed.includes(id) })),
    })),
    m = citationMetrics(judged);
  function toggle(i: number, id: string) {
    setChosen(
      chosen.map((row, j) =>
        j !== i
          ? row
          : row.includes(id)
            ? row.filter((v) => v !== id)
            : [...row, id],
      ),
    );
  }
  return (
    <section
      className="reference-lab workflow-lab"
      aria-label="Citation support calculator"
    >
      <p className="workflow-kicker">
        Calculated exercise · explicit human support judgments
      </p>
      <button onClick={() => setChosen([["hours"], ["loans"], ["birch"]])}>
        Reset experiment
      </button>
      <div className="workflow-stats" aria-live="polite">
        <Stat
          label="Citation precision"
          value={percent(m.citation_precision)}
        />
        <Stat label="Citation recall" value={percent(m.citation_recall)} />
      </div>
      {claims.map((c, i) => (
        <fieldset key={i}>
          <legend>{c.text}</legend>
          <div className="workflow-path">
            {["hours", "wren", "birch", "loans"].map((id) => (
              <label className="retrieval-check" key={id}>
                <input
                  type="checkbox"
                  checked={chosen[i].includes(id)}
                  onChange={() => toggle(i, id)}
                />
                {id}
              </label>
            ))}
          </div>
          <p>
            Jointly supported: {String(judged[i].fully_supported_by_citations)}.{" "}
            {i === 2
              ? "This claim needs both the Wren→Birch relationship and the Birch→42 contact."
              : ""}
          </p>
        </fieldset>
      ))}
      <details>
        <summary>Inspect source evidence</summary>
        {data.retrieval.documents
          .filter((d: any) =>
            ["hours", "wren", "birch", "loans"].includes(d.id),
          )
          .map((d: any) => (
            <p key={d.id}>
              <strong>{d.id}:</strong> {d.text}
            </p>
          ))}
      </details>
      <p>
        Precision judges whether each cited passage contributes support. Recall
        asks whether each complete claim is supported by its citation set. A
        partial bridge citation can be useful without fully supporting the whole
        claim. This calculator uses declared human judgments, not an automated
        entailment model.
      </p>
    </section>
  );
}
export function VisionExplorer() {
  const v = data.vision,
    [image, setImage] = useState(0),
    [wording, setWording] = useState("descriptive"),
    [enabled, setEnabled] = useState([0, 1, 2, 3, 4]);
  const row = v.images[image],
    bare = wording === "bare",
    labels = bare ? ["astronaut", "cat", "coffee"] : v.descriptions,
    indices = enabled.filter((i) => i < labels.length),
    cosines = (bare ? v.bare_cosines : v.cosines)[image],
    prob = indices.length
      ? candidateSoftmax(
          indices.map((i) => cosines[i]),
          v.logit_scale,
        )
      : [];
  function toggle(i: number) {
    setEnabled(
      enabled.includes(i)
        ? enabled.filter((v) => v !== i)
        : [...enabled, i].sort((a, b) => a - b),
    );
  }
  return (
    <section
      className="reference-lab workflow-lab"
      aria-label="Real image and text retrieval"
    >
      <p className="workflow-kicker">
        L14a · actual OpenCLIP vectors · recorded inference
      </p>
      <button
        onClick={() => {
          setImage(0);
          setWording("descriptive");
          setEnabled([0, 1, 2, 3, 4]);
        }}
      >
        Reset experiment
      </button>
      <div className="reference-controls">
        <label>
          Image
          <select
            aria-label="Vision image"
            value={image}
            onChange={(e) => setImage(Number(e.target.value))}
          >
            {v.images.map((r: any, i: number) => (
              <option value={i} key={r.id}>
                {r.id}
              </option>
            ))}
          </select>
        </label>
        <label>
          Recorded text wording
          <select
            aria-label="Vision wording"
            value={wording}
            onChange={(e) => setWording(e.target.value)}
          >
            <option value="descriptive">
              Descriptive prompts + distractors
            </option>
            <option value="bare">Bare class names</option>
          </select>
        </label>
      </div>
      <div className="vision-comparison">
        <figure>
          <img
            src={base + "/" + row.path.replace(/^public\//, "")}
            alt={row.alt}
            loading="lazy"
          />
          <figcaption>Original {row.original_size.join(" × ")}</figcaption>
        </figure>
        <figure>
          <img
            src={base + row.display_input}
            alt={"Inverse-normalized actual OpenCLIP input for " + row.id}
            loading="lazy"
          />
          <figcaption>Actual model view · 224 × 224</figcaption>
        </figure>
      </div>
      <p>
        {percent(row.area_retained)} of original area remains after center crop.
        Float tensor shape [{row.processed_shape.join(", ")}], {row.dtype}.
        Preview reverses normalization and rounds pixels; exact normalized
        tensor values are used for inference.
      </p>
      <div className="workflow-path">
        {labels.map((s: string, i: number) => (
          <label className="retrieval-check" key={i}>
            <input
              type="checkbox"
              checked={enabled.includes(i)}
              onChange={() => toggle(i)}
            />
            {s}
          </label>
        ))}
      </div>
      <div className="workflow-scroll">
        <table>
          <thead>
            <tr>
              <th>Candidate text</th>
              <th>Actual cosine</th>
              <th>Candidate-set softmax</th>
            </tr>
          </thead>
          <tbody>
            {indices.map((i, j) => (
              <tr key={i}>
                <td>{labels[i]}</td>
                <td>{cosines[i].toFixed(4)}</td>
                <td>{percent(prob[j])}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {!indices.length && (
        <p role="status">Select at least one candidate to calculate softmax.</p>
      )}
      <p>
        Removing a candidate recalculates the displayed softmax only. It does
        not rerun either encoder or change any cosine. These are relative scores
        among selected captions, not calibrated probabilities that an object
        exists.
      </p>
      <details>
        <summary>Normalization, actual text IDs, and provenance</summary>
        <Json
          value={{
            mean: row.mean,
            std: row.std,
            range: row.range,
            source_crop_box: row.source_crop_box,
            embedding_shapes: v.embedding_shapes,
            token_ids: bare
              ? "The bare-name condition is separately encoded in the source."
              : v.token_ids,
            model: v.models,
          }}
        />
        <p>
          {row.credit} · {row.license}.{" "}
          <a href={row.license_evidence}>Attribution evidence</a>.
        </p>
      </details>
    </section>
  );
}
export function PatchExplorer() {
  const v = data.vision,
    [selected, setSelected] = useState(0),
    [scale, setScale] = useState(v.logit_scale);
  const patch = v.patch_trace.patches[selected],
    loss = contrastiveLoss(
      v.cosines.map((r: number[]) => r.slice(0, 3).map((n) => n * scale)),
    );
  return (
    <section
      className="reference-lab workflow-lab"
      aria-label="Actual image patch projection"
    >
      <p className="workflow-kicker">
        Actual astronaut input · OpenCLIP ViT-B/32
      </p>
      <button
        onClick={() => {
          setSelected(0);
          setScale(v.logit_scale);
        }}
      >
        Reset experiment
      </button>
      <div className="patch-frame">
        <img
          src={base + v.images[0].display_input}
          alt="Actual astronaut model input divided into 49 patches"
        />
        <div className="patch-grid">
          {v.patch_trace.patches.map((p: any) => (
            <button
              key={p.index}
              aria-label={`Inspect patch row ${p.row + 1} column ${p.column + 1}`}
              aria-pressed={p.index === selected}
              onClick={() => setSelected(p.index)}
            >
              {p.index + 1}
            </button>
          ))}
        </div>
      </div>
      <p>
        One patch: [3,32,32] → 3072 flattened numbers → 768 learned projection
        values. Add a learned image class token: 49 + 1 = 50 tokens, each with a
        768-dimensional positional vector. This 7×7 grid belongs to OpenCLIP,
        not BLIP-2’s different vision encoder.
      </p>
      <Json value={patch} />
      <label>
        Counterfactual contrastive logit scale: {scale.toFixed(1)}
        <input
          aria-label="Contrastive logit scale"
          type="range"
          min="1"
          max="100"
          step="1"
          value={scale}
          onChange={(e) => setScale(Number(e.target.value))}
        />
      </label>
      <div className="workflow-stats" aria-live="polite">
        <Stat label="Image → text loss" value={loss.image_to_text.toFixed(5)} />
        <Stat label="Text → image loss" value={loss.text_to_image.toFixed(5)} />
        <Stat label="Symmetric mean loss" value={loss.loss.toFixed(5)} />
      </div>
      <p>
        Loss is recalculated from the three actual paired frozen embeddings.
        Changing this calculation does not train the model. The Python source
        verifies the arithmetic against PyTorch cross entropy and verifies patch
        projection against the actual convolution.
      </p>
    </section>
  );
}
export function VisualGeneration({ mode = "caption" }: { mode?: "caption" | "chat" }) {
  const [backend, setBackend] = useState<string>(mode),
    [index, setIndex] = useState(0),
    d = data[backend];
  if (!d) return <p>Recorded visual generation is awaiting verification.</p>;
  const r = d.rows[Math.min(index, d.rows.length - 1)];
  return (
    <section
      className="reference-lab workflow-lab"
      aria-label="Actual visual generation"
    >
      <p className="workflow-kicker">
        L14 · actual pretrained generation · no browser model execution
      </p>
      <button
        onClick={() => {
          setBackend(mode);
          setIndex(0);
        }}
      >
        Reset experiment
      </button>
      <div className="reference-controls">
        <label>
          Verified backend
          <select
            aria-label="Visual generation backend"
            value={backend}
            onChange={(e) => {
              setBackend(e.target.value);
              setIndex(0);
            }}
          >
            {data.caption && (
              <option value="caption">Genuine BLIP-2 · Linux CPU</option>
            )}
            {data.chat && (
              <option value="chat">SmolVLM · small CPU chat</option>
            )}
          </select>
        </label>
        <label>
          Case
          <select
            aria-label="Visual generation case"
            value={Math.min(index, d.rows.length - 1)}
            onChange={(e) => setIndex(Number(e.target.value))}
          >
            {d.rows.map((r: any, i: number) => (
              <option value={i} key={i}>
                {backend === "caption" ? r.image_id : `${r.condition ?? "chat"} · Turn ${r.turn + 1}`}
              </option>
            ))}
          </select>
        </label>
      </div>
      <img
        className="vision-generation-image"
        src={`${base}/images/e6/${r.image_id ?? d.image_id}.png`}
        alt={r.image_id ?? "Coffee photograph used in both chat turns"}
      />
      <p>
        <strong>
          {r.question || r.prompt || "Caption without an added text question"}
        </strong>
      </p>
      <pre className="workflow-pre">{r.raw}</pre>
      {backend === "caption" ? (
        <>
          <div className="retrieval-bridge">
            {Object.entries(r.trace).map(([key, value]) => (
              <div key={key}>
                <strong>{key.replaceAll("_", " ")}</strong>
                <code>{JSON.stringify(value)}</code>
              </div>
            ))}
          </div>
          <p>
            Actual vision → query → language projection shapes. The original
            Q-Former queries and projection were learned during pretraining;
            this lab performs inference only.
          </p>
          <details>
            <summary>Verified hardware placement and input view</summary>
            <img
              className="vision-generation-image"
              src={base + r.display_input}
              alt="Actual inverse-normalized BLIP-2 pixel input"
            />
            <Json
              value={{ profile: d.profile, bridge_config: d.bridge_config }}
            />
          </details>
        </>
      ) : (
        <>
          <p>
            First-turn conditions change image preprocessing alone; second turns also carry the changed generated history. The second turn contains the first actual generated answer. An exact
            drink price is not visible in the photograph; a confident amount is
            unsupported.
          </p>
          <details>
            <summary>
              Actual conversation and serialized multimodal prompt
            </summary>
            <Json value={r.messages} />
            <pre className="workflow-pre">{r.rendered_prompt}</pre>
          </details>
        </>
      )}
      <details>
        <summary>Actual tensor shapes, dtypes, and token evidence</summary>
        <Json
          value={{
            shapes: r.input_shapes,
            dtypes: r.dtypes,
            tokens: r.input_ids ?? r.input_tokens,
            text_tokens: r.text_tokens,
            settings: r.settings ?? { max_new_tokens: r.max_new_tokens },
          }}
        />
      </details>
    </section>
  );
}
