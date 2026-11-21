import { useState } from "react";
import recorded from "../../data/workflow-results.json";
import {
  aggregateAnswers,
  validateOrder,
} from "../../lib/workflow-mechanics.js";
// Published JSON is hash-linked to actual local runs. Controls select evidence; no browser model calls.
const data: Record<string, any> = recorded;
const pct = (n: number) => (n * 100).toFixed(1) + "%";
function Json({ value }: { value: unknown }) {
  return <pre className="workflow-pre">{JSON.stringify(value, null, 2)}</pre>;
}
function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="workflow-stat">
      <strong>{value}</strong>
      <span>{label}</span>
    </div>
  );
}
function Call({
  call,
  label = "Actual model call",
}: {
  call: any;
  label?: string;
}) {
  return (
    <details className="workflow-call">
      <summary>
        {label} · {call.prompt_tokens} input / {call.output_tokens} output
        tokens · {call.finish_reason}
      </summary>
      <p>Unedited output</p>
      <pre className="workflow-pre">{call.raw || "(empty)"}</pre>
      <details>
        <summary>Typed messages, serialized prompt, and settings</summary>
        <Json value={call.messages} />
        <pre className="workflow-pre">{call.rendered_prompt}</pre>
        <Json value={call.settings} />
      </details>
    </details>
  );
}
export function PromptExperiment({ initial = "direct" }: { initial?: string }) {
  const [variant, setVariant] = useState(initial),
    [index, setIndex] = useState(0);
  const p = data.prompts,
    all = [...p.arithmetic, ...p.extraction],
    variants = p.summary.map((s: any) => s.variant),
    rows = all.filter((r) => r.variant === variant),
    row = rows[Math.min(index, rows.length - 1)],
    stats = p.summary.find((s: any) => s.variant === variant);
  return (
    <section
      className="reference-lab workflow-lab"
      aria-label="Recorded prompt experiment"
    >
      <p className="workflow-kicker">
        L09 · recorded local model · held-out tasks
      </p>
      <button
        onClick={() => {
          setVariant(initial);
          setIndex(0);
        }}
      >
        Reset experiment
      </button>
      <div className="reference-controls">
        <label>
          Prompt variant
          <select
            value={variant}
            onChange={(e) => {
              setVariant(e.target.value);
              setIndex(0);
            }}
          >
            {variants.map((v: string) => (
              <option key={v}>{v}</option>
            ))}
          </select>
        </label>
        <label>
          Evaluation case
          <select
            value={Math.min(index, rows.length - 1)}
            onChange={(e) => setIndex(Number(e.target.value))}
          >
            {rows.map((r: any, i: number) => (
              <option value={i} key={r.id}>
                {r.id}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="workflow-stats" aria-live="polite" aria-atomic="true">
        <Stat
          label="Extracted-answer accuracy"
          value={pct(stats.answer_accuracy)}
        />
        <Stat
          label="Requested-format validity"
          value={pct(stats.format_validity)}
        />
        <Stat label="Cases in this condition" value={stats.cases} />
      </div>
      <p>{row.question ?? row.text}</p>
      <div className="workflow-comparison">
        <div>
          <strong>Reference</strong>
          <Json value={row.expected} />
        </div>
        <div>
          <strong>
            Generated ·{" "}
            {row.answer_correct ? "correct" : "incorrect / unextractable"}
          </strong>
          <pre className="workflow-pre">{row.call.raw}</pre>
        </div>
      </div>
      <p>
        Example IDs: {row.example_ids.join(", ") || "none"}. Format:{" "}
        {String(row.format_valid ?? row.validation.valid)}. Parsed:{" "}
        {JSON.stringify(row.normalized_answer ?? row.validation?.value ?? null)}
        .
      </p>
      <Call call={row.call} label="Inspect actual input and generation" />
      <details>
        <summary>Compare every predeclared condition</summary>
        <div className="workflow-scroll">
          <table>
            <thead>
              <tr>
                <th>Condition</th>
                <th>Answer</th>
                <th>Format</th>
                <th>Input/output tokens</th>
              </tr>
            </thead>
            <tbody>
              {p.summary.map((s: any) => (
                <tr key={s.variant}>
                  <td>{s.variant}</td>
                  <td>{pct(s.answer_accuracy)}</td>
                  <td>{pct(s.format_validity)}</td>
                  <td>
                    {s.prompt_tokens} / {s.output_tokens}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p>
          Arithmetic uses 8 evaluation cases; extraction uses 9. These are
          distinct tasks. Accuracy uses the declared strict parser, so
          unextractable prose scores zero even when a human could infer its
          answer. Single sequential timings use a reused runtime/prefix cache
          and are not a speed ranking.
        </p>
      </details>
      <details>
        <summary>One-token native truncation probe</summary>
        <Call call={p.truncation_probe} />
        <p>
          The prefix is grammar-allowed but incomplete; validation fails. Native
          decoding is not post-generation repair.
        </p>
      </details>
    </section>
  );
}
export function VotingLab() {
  const [selected, setSelected] = useState(0),
    [custom, setCustom] = useState("15, 15, 14, invalid, invalid");
  const s = data.search.samples[selected];
  const parsed = custom
      .split(",")
      .map((s) => (/^-?\d+$/.test(s.trim()) ? Number(s.trim()) : null)),
    vote = aggregateAnswers(parsed);
  return (
    <section
      className="reference-lab workflow-lab"
      aria-label="Sampled answer voting"
    >
      <p className="workflow-kicker">L10 · five actual samples per condition</p>
      <button
        onClick={() => {
          setSelected(0);
          setCustom("15, 15, 14, invalid, invalid");
        }}
      >
        Reset experiment
      </button>
      <label>
        Recorded case and mechanism
        <select
          value={selected}
          onChange={(e) => setSelected(Number(e.target.value))}
        >
          {data.search.samples.map((r: any, i: number) => (
            <option value={i} key={i}>
              {r.id} · {r.variant}
            </option>
          ))}
        </select>
      </label>
      <p>
        {s.question} Reference: <strong>{s.expected}</strong>.
      </p>
      <div className="workflow-stats" aria-live="polite" aria-atomic="true">
        <Stat
          label="Unique plurality"
          value={s.aggregation.winner ?? "abstain"}
        />
        <Stat label="Invalid samples" value={s.aggregation.invalid} />
        <Stat
          label="Strict majority"
          value={String(s.aggregation.strict_majority)}
        />
      </div>
      <ol>
        {s.samples.map((r: any, i: number) => (
          <li key={i}>
            <strong>
              Sample {i + 1} → {r.answer ?? "unextractable"} ·{" "}
              {r.correct ? "correct" : "incorrect"}
            </strong>
            <Call call={r.call} />
          </li>
        ))}
      </ol>
      <p>
        The two paths change both instruction and decoding; they are mechanism
        probes, not an isolated comparison of one intervention. Repeated
        agreement can still be wrong.
      </p>
      <details open>
        <summary>Counterfactual vote calculator · no model execution</summary>
        <label>
          Comma-separated answers; other text counts as invalid
          <input value={custom} onChange={(e) => setCustom(e.target.value)} />
        </label>
        <Json value={vote} />
        <p>
          Ties abstain. Strict majority requires more than half of all samples,
          including invalids.
        </p>
      </details>
    </section>
  );
}
export function SearchExplorer() {
  const [caseId, setCase] = useState("game-simple"),
    [strategy, setStrategy] = useState("complete"),
    [nodeId, setNode] = useState(0);
  const cases = [
    ...new Set(data.search.games.map((r: any) => r.id)),
  ] as string[];
  const run =
    strategy === "model-guided"
      ? data.search.guided
      : data.search.games.find(
          (r: any) => r.id === caseId && r.strategy === strategy,
        );
  const nodes = run.display_nodes,
    current = nodes.find((n: any) => n.id === nodeId) ?? nodes[0];
  const route = run.solution_path
    .map((id: number) => nodes.find((n: any) => n.id === id))
    .filter(Boolean);
  const visible = nodes
    .filter((n: any) => n.depth < 2 || run.solution_path.includes(n.id))
    .slice(0, 55);
  const layout = new Map<number, { x: number; y: number }>();
  for (let depth = 0; depth < 4; depth++) {
    const level = visible.filter((n: any) => n.depth === depth);
    level.forEach((n: any, i: number) =>
      layout.set(n.id, {
        x: ((i + 1) * 900) / (level.length + 1),
        y: 35 + depth * 74,
      }),
    );
  }
  return (
    <section
      className="reference-lab workflow-lab"
      aria-label="Actual bounded search tree"
    >
      <p className="workflow-kicker">
        L10 · exact rational search · actual node trace
      </p>
      <button
        onClick={() => {
          setCase("game-simple");
          setStrategy("complete");
          setNode(0);
        }}
      >
        Reset experiment
      </button>
      <div className="reference-controls">
        <label>
          Puzzle
          <select
            value={caseId}
            onChange={(e) => {
              setCase(e.target.value);
              setStrategy("complete");
              setNode(0);
            }}
          >
            {cases.map((c) => (
              <option key={c}>{c}</option>
            ))}
          </select>
        </label>
        <label>
          Frontier policy
          <select
            value={strategy}
            onChange={(e) => {
              setStrategy(e.target.value);
              setNode(0);
            }}
          >
            {[
              "complete",
              "beam-1",
              "beam-4",
              "budget-16",
              ...(caseId === "game-simple" ? ["model-guided"] : []),
            ].map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
        </label>
      </div>
      <p>
        Use {run.numbers.join(", ")} exactly once to make 24. Stop:{" "}
        <strong>{run.stop_reason}</strong>.{" "}
        {run.solution ||
          "No solution found under this run’s policy and budget."}
      </p>
      <div className="workflow-stats" aria-live="polite" aria-atomic="true">
        <Stat label="Nodes created" value={run.created} />
        <Stat label="Expanded" value={run.expanded} />
        <Stat label="Beam pruned" value={run.beam_pruned} />
        <Stat
          label="Exact verification"
          value={run.verified ? "passed" : "no solution"}
        />
      </div>
      <div
        className="workflow-tree"
        tabIndex={0}
        aria-label="Search overview; select nodes using the buttons below"
      >
        <svg
          viewBox="0 0 900 300"
          role="img"
          aria-label="Actual root, first branch level, and solution ancestors"
        >
          <title>Recorded search branches</title>
          {visible.map((n: any) => {
            const pos = layout.get(n.id)!,
              parent = layout.get(n.parent);
            return (
              <g key={n.id}>
                {parent && (
                  <line x1={parent.x} y1={parent.y} x2={pos.x} y2={pos.y} />
                )}
                <circle
                  cx={pos.x}
                  cy={pos.y}
                  r={run.solution_path.includes(n.id) ? 9 : 5}
                  className={
                    n.status === "beam_pruned" ? "workflow-pruned" : ""
                  }
                />
                {(n.depth === 0 || run.solution_path.includes(n.id)) && (
                  <text x={pos.x} y={pos.y + 23} textAnchor="middle">
                    #{n.id}
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      </div>
      <p>
        Overview shows root, first-level alternatives, and solution ancestors.
        Gray nodes were pruned. Select any published node below to inspect its
        actual state.
      </p>
      <label>
        Inspect node
        <select
          value={current.id}
          onChange={(e) => setNode(Number(e.target.value))}
        >
          {nodes.map((n: any) => (
            <option key={n.id} value={n.id}>
              #{n.id} · depth {n.depth} · {n.action} · {n.status}
            </option>
          ))}
        </select>
      </label>
      <Json value={current} />
      <div className="workflow-path">
        {current.parent !== null && (
          <button onClick={() => setNode(current.parent)}>
            ← Parent #{current.parent}
          </button>
        )}
        <details>
          <summary>Explore child branches of node #{current.id}</summary>
          <div className="workflow-path">
            {nodes
              .filter((n: any) => n.parent === current.id)
              .map((n: any) => (
                <button key={n.id} onClick={() => setNode(n.id)}>
                  #{n.id}: {n.action}
                </button>
              ))}
          </div>
        </details>
      </div>
      {route.length > 0 && (
        <details open>
          <summary>Verified solution route</summary>
          <ol>
            {route.map((n: any) => (
              <li key={n.id}>
                <button onClick={() => setNode(n.id)}>{n.action}</button> →{" "}
                {n.values.join(", ")}
              </li>
            ))}
          </ol>
        </details>
      )}
      <p>
        Published trace: {nodes.length} nodes; truncated:{" "}
        {String(run.display_truncated)}. The full local trace is saved by the
        notebook. Beam failure and budget exhaustion do not prove impossibility.
      </p>
      {strategy === "model-guided" && (
        <details>
          <summary>Actual model candidate-ranking calls</summary>
          <p>
            The model chooses one of at most 8 legal candidates. The solver
            constructs and verifies arithmetic. At most 3 model calls.
          </p>
          {data.search.model_choices.map((r: any, i: number) => (
            <div key={i}>
              <Json value={{ options: r.options, selected: r.selected }} />
              <Call call={r.call} />
            </div>
          ))}
        </details>
      )}
    </section>
  );
}
export function MemoryExplorer({ initial = "buffer" }: { initial?: string }) {
  const [policy, setPolicy] = useState(initial),
    [budget, setBudget] = useState(384);
  const m = data.memory,
    r = m.cases.find((r: any) => r.policy === policy && r.budget === budget),
    p = r.packed;
  return (
    <section
      className="reference-lab workflow-lab"
      aria-label="Memory policy and context budget"
    >
      <p className="workflow-kicker">
        L11 · same conversation · actual answers
      </p>
      <button
        onClick={() => {
          setPolicy(initial);
          setBudget(384);
        }}
      >
        Reset experiment
      </button>
      <div className="reference-controls">
        <label>
          Memory policy
          <select value={policy} onChange={(e) => setPolicy(e.target.value)}>
            {["buffer", "window", "summary"].map((p) => (
              <option key={p}>{p}</option>
            ))}
          </select>
        </label>
        <label>
          Application token budget
          <select
            value={budget}
            onChange={(e) => setBudget(Number(e.target.value))}
          >
            <option>192</option>
            <option>384</option>
          </select>
        </label>
      </div>
      <div className="workflow-stats" aria-live="polite" aria-atomic="true">
        <Stat label="Actual input tokens" value={p.input_tokens} />
        <Stat label="Reserved output tokens" value={p.reserved_output_tokens} />
        <Stat label="Summary retained" value={String(p.summary_kept)} />
        <Stat
          label="Fields correct"
          value={`${Object.values(r.field_correct).filter(Boolean).length} / 3`}
        />
      </div>
      <meter
        min={0}
        max={budget}
        value={p.input_tokens + p.reserved_output_tokens}
        aria-label="Input plus reserved output within application budget"
      />
      <p>
        {p.input_tokens} input + {p.reserved_output_tokens} output reserve ≤{" "}
        {budget}; native context capacity is separately 768.
      </p>
      <div className="workflow-comparison">
        <div>
          <strong>Reference facts</strong>
          <Json value={m.expected_facts} />
        </div>
        <div>
          <strong>Generated answer</strong>
          <pre className="workflow-pre">{r.call.raw}</pre>
        </div>
      </div>
      <Json
        value={{
          retained: p.retained_ids,
          policy_omissions: p.omitted_by_policy,
          budget_evictions: p.dropped_for_budget,
          field_correct: r.field_correct,
          literal_presence: r.literal_evidence_presence,
        }}
      />
      <Call call={r.call} />
      <details>
        <summary>
          Original conversation with retained-message highlights
        </summary>
        <ol>
          {m.history.map((h: any) => (
            <li
              key={h.id}
              className={
                p.retained_ids.includes(h.id) ? "workflow-retained" : ""
              }
            >
              <strong>
                {h.id} · {h.role}
              </strong>
              : {h.content}{" "}
              {p.retained_ids.includes(h.id) ? "✓ retained verbatim" : ""}
            </li>
          ))}
        </ol>
      </details>
      <details>
        <summary>Actual running-summary updates</summary>
        {m.summary_updates.map((u: any, i: number) => (
          <div key={i}>
            <p>
              Update {i + 1}; new IDs: {u.new_message_ids.join(", ")}
            </p>
            <Call call={u.call} />
          </div>
        ))}
        <p>
          These generated notes can omit facts or invent information. Literal
          presence does not prove correct use.
        </p>
      </details>
    </section>
  );
}
export function WorkflowTrace({ initial = "chain" }: { initial?: string }) {
  const [mode, setMode] = useState(initial),
    [index, setIndex] = useState(0),
    [engine, setEngine] = useState("plain");
  const a = data.agents,
    c = data.chains;
  const rows =
      mode === "faketools"
        ? a.fake_cases
        : a.real_cases.filter(
            (r: any) =>
              r.engine === (engine === "framework" ? "langchain" : engine),
          ),
    row = rows[Math.min(index, rows.length - 1)],
    outcome = mode === "faketools" ? row[engine] : row.outcome;
  return (
    <section
      className="reference-lab workflow-lab"
      aria-label="Local model and tool workflow trace"
    >
      <p className="workflow-kicker">
        L11 · real calls and explicitly labeled fake tests
      </p>
      <button
        onClick={() => {
          setMode(initial);
          setIndex(0);
          setEngine("plain");
        }}
      >
        Reset experiment
      </button>
      <label>
        Explore a workflow
        <select
          value={mode}
          onChange={(e) => {
            setMode(e.target.value);
            setIndex(0);
          }}
        >
          <option value="runtime">GGUF runtime and equivalent templates</option>
          <option value="chain">Actual two-call chain</option>
          <option value="realtools">Actual model-controlled tools</option>
          <option value="faketools">
            Scripted / FakeListLLM controller tests
          </option>
        </select>
      </label>
      {mode === "runtime" ? (
        <>
          <Json value={c.model} />
          <p>
            Plain/framework serialized prompts match:{" "}
            {String(c.equivalent_prompt_checked)}. Greedy outputs match:{" "}
            {String(c.equivalent_output_checked)}. Embedded template checked:{" "}
            {String(c.gguf_template_checked)}. Q8 file type checked:{" "}
            {String(c.quantization_type_checked)}.
          </p>
          <Call call={c.plain} label="Plain-Python path" />
          <Call call={c.framework} label="Current LangChain Runnable path" />
        </>
      ) : mode === "chain" ? (
        <>
          <p>Request: {c.multi_prompt.request}</p>
          <Call
            call={c.multi_prompt.first_call}
            label="1 · Actual extraction call"
          />
          <Json value={c.multi_prompt.state} />
          <Call
            call={c.multi_prompt.second_call}
            label="2 · Actual copy-total call"
          />
          <Json
            value={Object.fromEntries(
              Object.entries(c.multi_prompt).filter(
                ([k]) =>
                  !["first_call", "second_call", "state", "request"].includes(
                    k,
                  ),
              ),
            )}
          />
          <p>
            Extraction, arithmetic for the extracted value, copy consistency,
            and end-to-end correctness are different checks.
          </p>
        </>
      ) : (
        <>
          <div className="reference-controls">
            <label>
              Execution path
              <select
                value={engine}
                onChange={(e) => {
                  setEngine(e.target.value);
                  setIndex(0);
                }}
              >
                <option value="plain">Plain Python</option>
                <option value="framework">LangChain core</option>
              </select>
            </label>
            <label>
              Case
              <select
                value={Math.min(index, rows.length - 1)}
                onChange={(e) => setIndex(Number(e.target.value))}
              >
                {rows.map((r: any, i: number) => (
                  <option key={r.id} value={i}>
                    {r.id}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <p>
            {mode === "faketools"
              ? "FAKE MODEL: scripted decisions test controller mechanics; no competence claim."
              : row.question}
          </p>
          <p>
            Stop: <strong>{outcome.stop_reason}</strong>. Answer:{" "}
            {outcome.answer ?? "none"}.{" "}
            {mode === "realtools" &&
              `Reference: ${row.expected}; correct: ${row.correct}.`}
          </p>
          <ol>
            {outcome.trace.map((t: any, i: number) => (
              <li key={i}>
                <strong>
                  Step {t.step} · {t.kind}
                </strong>
                <Json value={t} />
                {row.decisions?.[i] && (
                  <>
                    <p>
                      Model choice ID: {row.decisions[i].selected_id}. Resolved
                      action:
                    </p>
                    <Json value={row.decisions[i].resolved_action} />
                    <Call call={row.decisions[i].call} />
                  </>
                )}
              </li>
            ))}
          </ol>
          <details>
            <summary>Controller boundaries</summary>
            <Json value={a.policy} />
            <p>
              Predeclared operands and finite choices; no arbitrary code
              execution, browsing, or write tools. The local model chooses among
              legal calls and can still choose incorrectly.
            </p>
          </details>
        </>
      )}
    </section>
  );
}
export function OutputValidator() {
  const [raw, setRaw] = useState(
    '{"item":"pen","quantity":8,"priority":"urgent"}',
  );
  const v = validateOrder(raw),
    correct =
      v.valid &&
      v.value?.item === "pen" &&
      v.value.quantity === 3 &&
      v.value.priority === "urgent";
  return (
    <section
      className="reference-lab workflow-lab"
      aria-label="Strict order validator"
    >
      <p className="workflow-kicker">
        Deterministic contract exercise · no model calls
      </p>
      <button
        onClick={() =>
          setRaw('{"item":"pen","quantity":8,"priority":"urgent"}')
        }
      >
        Reset experiment
      </button>
      <p>
        Request: Please send <strong>3 pens</strong> with{" "}
        <strong>urgent</strong> delivery.
      </p>
      <label>
        Candidate JSON
        <textarea
          value={raw}
          onChange={(e) => setRaw(e.target.value)}
          rows={4}
          spellCheck={false}
        />
      </label>
      <div className="workflow-stats" aria-live="polite" aria-atomic="true">
        <Stat label="Contract valid" value={String(v.valid)} />
        <Stat label="Content correct" value={String(correct)} />
      </div>
      {v.error && <p role="status">{v.error}</p>}
      <div className="workflow-path">
        <button
          onClick={() =>
            setRaw('{"item":"pen","quantity":3,"priority":"urgent"}')
          }
        >
          Correct object
        </button>
        <button
          onClick={() =>
            setRaw('{"item":"pen","quantity":true,"priority":"urgent"}')
          }
        >
          Boolean quantity
        </button>
        <button
          onClick={() =>
            setRaw(
              '{"item":"pen","quantity":3,"quantity":8,"priority":"urgent"}',
            )
          }
        >
          Duplicate key
        </button>
        <button
          onClick={() =>
            setRaw('{"item":"pen","quantity":3.0,"priority":"urgent"}')
          }
        >
          Decimal quantity
        </button>
      </div>
      <p>
        Exactly three fields; item pen/book, priority normal/urgent, integer
        quantity 1–9. Duplicate keys, extra prose, booleans, decimals, and extra
        fields fail. Shape validity and source correctness stay separate.
      </p>
    </section>
  );
}
