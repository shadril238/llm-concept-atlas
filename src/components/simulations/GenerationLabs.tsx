import { useState } from "react";
import published from "../../data/generation-results.json";
import { preferenceLoss, lossMask } from "../../lib/generation-mechanics.js";
const data: Record<string, any> = published;
const num = (x: number) => x.toLocaleString("en-US");
const Json = ({ value }: { value: unknown }) => (
  <pre className="workflow-pre">{JSON.stringify(value, null, 2)}</pre>
);

export function InstructionMask() {
  const [index, setIndex] = useState(0),
    [mode, setMode] = useState("completion");
  const row = data.sft.inspection[index],
    ids = [...row.input_ids, 0, 0],
    mask = lossMask(row.input_ids, row.prefix_tokens, 2, mode);
  return (
    <section
      className="workflow-lab generation-lab"
      aria-label="Instruction loss mask inspector"
    >
      <p className="workflow-eyebrow">
        Inspect actual tokenizer output · calculate a mask
      </p>
      <h3>Which tokens teach the model?</h3>
      <div className="workflow-controls">
        <label>
          Inspection case
          <select value={index} onChange={(e) => setIndex(+e.target.value)}>
            {data.sft.inspection.map((r: any, i: number) => (
              <option key={r.id} value={i}>
                {r.id}
              </option>
            ))}
          </select>
        </label>
        <label>
          Supervision policy
          <select value={mode} onChange={(e) => setMode(e.target.value)}>
            <option value="completion">Completion only · trained policy</option>
            <option value="all">All real tokens · comparison</option>
          </select>
        </label>
        <button
          onClick={() => {
            setIndex(0);
            setMode("completion");
          }}
        >
          Reset mask
        </button>
      </div>
      <p aria-live="polite">
        {mask.filter((x) => x !== -100).length} labels contribute before causal
        shifting. Prompt tokens{" "}
        {mode === "completion" ? "are excluded" : "also receive loss"}. The real
        EOS is included; the two appended padding positions are excluded
        although their ID is also zero.
      </p>
      <div className="workflow-scroll generation-token-scroll">
        <table>
          <thead>
            <tr>
              <th>Position</th>
              <th>Token</th>
              <th>ID</th>
              <th>Label</th>
              <th>Role</th>
            </tr>
          </thead>
          <tbody>
            {ids.map((id: number, i: number) => (
              <tr
                key={i}
                className={mask[i] === -100 ? "" : "generation-target"}
              >
                <td>{i}</td>
                <td>
                  <code>{row.tokens[i] ?? "[appended pad]"}</code>
                </td>
                <td>{id}</td>
                <td>{mask[i] === -100 ? "−100 · ignored" : mask[i]}</td>
                <td>
                  {i >= row.input_ids.length
                    ? "padding"
                    : i < row.prefix_tokens
                      ? "prompt"
                      : i === row.input_ids.length - 1
                        ? "real EOS"
                        : "completion"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p>
        Labels shift inside the causal loss: logit at position t predicts label
        t+1. Pythia uses ordinary text delimiters here, not native chat control
        tokens.
      </p>
    </section>
  );
}

export function AdaptationResults() {
  const [kind, setKind] = useState("lora"),
    [index, setIndex] = useState(0);
  const run = data.sft.runs.find((r: any) => r.kind === kind),
    before = data.sft.baseline.generations[index],
    after = run.generations[index];
  return (
    <section
      className="workflow-lab generation-lab"
      aria-label="Actual instruction training results"
    >
      <p className="workflow-eyebrow">
        Recorded optimization · heldout behavior
      </p>
      <h3>What changed after instruction tuning?</h3>
      <div className="workflow-controls">
        <label>
          Trained route
          <select value={kind} onChange={(e) => setKind(e.target.value)}>
            <option value="lora">LoRA</option>
            <option value="full">Full fine-tuning</option>
          </select>
        </label>
        <label>
          Heldout task
          <select value={index} onChange={(e) => setIndex(+e.target.value)}>
            {run.generations.map((r: any, i: number) => (
              <option key={r.id} value={i}>
                {r.id}
              </option>
            ))}
          </select>
        </label>
        <button
          onClick={() => {
            setKind("lora");
            setIndex(0);
          }}
        >
          Reset results
        </button>
      </div>
      <div className="workflow-stats">
        <div className="workflow-stat">
          <strong>{num(run.trainable)}</strong>
          <span>Trainable parameters</span>
        </div>
        <div className="workflow-stat">
          <strong>{run.selected_step}</strong>
          <span>Validation-selected trained step</span>
        </div>
        <div className="workflow-stat">
          <strong>{run.test_nll.toFixed(3)}</strong>
          <span>
            Heldout completion NLL; baseline {data.sft.baseline.nll.toFixed(3)}
          </span>
        </div>
      </div>
      <details>
        <summary>Validation curve and artifact checks</summary>
        <Json
          value={{
            trace: run.trace,
            reload: run.reload_max_abs_error,
            merge: run.merge_max_abs_error,
            float64_merge: run.float64_merge_max_abs_error,
            selection: run.selection,
          }}
        />
      </details>
      <div aria-live="polite">
        <h4>Instruction</h4>
        <p>{after.instruction}</p>
        <h4>Source context</h4>
        <p>{after.context}</p>
        <h4>Dataset reference</h4>
        <pre className="workflow-pre">{after.reference}</pre>
        <h4>Base output</h4>
        <pre className="workflow-pre">{before.prediction || "[empty]"}</pre>
        <h4>{kind} output</h4>
        <pre className="workflow-pre">{after.prediction || "[empty]"}</pre>
        <p>
          24-token generation limit reached: {String(after.hit_limit)}. All
          outputs are actual saved generations; the reference is not inserted
          into decoding.
        </p>
      </div>
    </section>
  );
}

export { RankMemory } from "./RankMemory";

export function PreferenceExplorer() {
  const [pc, setPc] = useState(-3),
    [pr, setPr] = useState(-5),
    [rc, setRc] = useState(-3),
    [rr, setRr] = useState(-5),
    [beta, setBeta] = useState(0.1),
    [index, setIndex] = useState(0);
  const result = preferenceLoss(pc, pr, rc, rr, beta),
    actual = data.preferences.dpo.test.rows[index],
    reward = data.preferences.reward.test.rows[index];
  return (
    <section
      className="workflow-lab generation-lab"
      aria-label="Reward and DPO preference trace"
    >
      <p className="workflow-eyebrow">
        Calculated DPO objective + separate recorded model scores
      </p>
      <h3>Compare policy and reference on the same completions</h3>
      <div className="workflow-controls">
        {[
          ["Policy chosen", pc, setPc],
          ["Policy rejected", pr, setPr],
          ["Reference chosen", rc, setRc],
          ["Reference rejected", rr, setRr],
        ].map(([name, value, setter]: any) => (
          <label key={name}>
            {name}: {value}
            <input
              type="range"
              min="-20"
              max="0"
              step=".1"
              value={value}
              onChange={(e) => setter(+e.target.value)}
            />
          </label>
        ))}
        <label>
          Beta: {beta}
          <input
            type="range"
            min=".05"
            max="1"
            step=".05"
            value={beta}
            onChange={(e) => setBeta(+e.target.value)}
          />
        </label>
        <button
          onClick={() => {
            setPc(-3);
            setPr(-5);
            setRc(-3);
            setRr(-5);
            setBeta(0.1);
            setIndex(0);
          }}
        >
          Reset preferences
        </button>
      </div>
      <p aria-live="polite">
        Reference-relative margin = {result.margin.toFixed(3)}; β × margin ={" "}
        {result.z.toFixed(3)}; loss = {result.loss.toFixed(5)}. Identical
        policy/reference gives log(2), with no relative winner. Beta is not the
        learning rate.
      </p>
      <label>
        Actual heldout pair
        <select value={index} onChange={(e) => setIndex(+e.target.value)}>
          {data.preferences.dpo.test.rows.map((r: any, i: number) => (
            <option key={r.id} value={i}>
              {r.id}
            </option>
          ))}
        </select>
      </label>
      <div aria-live="polite">
        <h4>Instruction</h4>
        <p>{reward.instruction}</p>
        <div className="generation-pair">
          <div>
            <strong>Chosen completion</strong>
            <p>{reward.chosen_text}</p>
          </div>
          <div>
            <strong>Rejected completion</strong>
            <p>{reward.rejected_text}</p>
          </div>
        </div>
        <div className="workflow-scroll">
          <table>
            <thead>
              <tr>
                <th>Scoring mechanism</th>
                <th>Chosen</th>
                <th>Rejected</th>
                <th>Difference</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <th>Scalar reward model</th>
                <td>{reward.chosen.toFixed(3)}</td>
                <td>{reward.rejected.toFixed(3)}</td>
                <td>{(reward.chosen - reward.rejected).toFixed(3)}</td>
              </tr>
              <tr>
                <th>Policy log probability</th>
                <td>{actual.policy_chosen.toFixed(3)}</td>
                <td>{actual.policy_rejected.toFixed(3)}</td>
                <td>
                  {(actual.policy_chosen - actual.policy_rejected).toFixed(3)}
                </td>
              </tr>
              <tr>
                <th>Reference log probability</th>
                <td>{actual.reference_chosen.toFixed(3)}</td>
                <td>{actual.reference_rejected.toFixed(3)}</td>
                <td>
                  {(
                    actual.reference_chosen - actual.reference_rejected
                  ).toFixed(3)}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <p>
          DPO reference-relative margin:{" "}
          <strong>{actual.relative_margin.toFixed(3)}</strong>. Defect:{" "}
          {reward.error_type}. Candidate lengths: {reward.chosen_tokens} /{" "}
          {reward.rejected_tokens} reward-tokenizer tokens.
        </p>
      </div>
      <details>
        <summary>Source context and exact recorded scores</summary>
        <Json
          value={{ reward_model: reward, dpo_policy_and_reference: actual }}
        />
      </details>
      <p>
        The scalar reward model was trained independently. DPO does not query
        it. Raw completion likelihood and reference-relative reward can
        disagree; both include length effects. Ties stay explicit.
      </p>
    </section>
  );
}

export function GenerationEvaluation() {
  const [model, setModel] = useState("lora"),
    [index, setIndex] = useState(0),
    [probe, setProbe] = useState("paraphrase");
  const run = data.evaluation.runs.find((r: any) => r.model === model),
    row = run.rows[index],
    p = data.evaluation.metric_probes.rows.find((r: any) => r.id === probe);
  return (
    <section
      className="workflow-lab generation-lab"
      aria-label="Generation metric disagreement dashboard"
    >
      <p className="workflow-eyebrow">
        Fixed suite · eight heldout source contexts
      </p>
      <h3>Read the answer behind the score</h3>
      <div className="workflow-scroll">
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Exact / 8</th>
              <th>ROUGE-L</th>
              <th>BLEU / 100</th>
              <th>Tiny BERTScore F1</th>
              <th>Completion PPL</th>
            </tr>
          </thead>
          <tbody>
            {data.evaluation.runs.map((r: any) => (
              <tr key={r.model}>
                <th>{r.model}</th>
                <td>{Math.round(r.exact_match * 8)}</td>
                <td>{r.rougeL.toFixed(3)}</td>
                <td>{r.bleu.toFixed(2)}</td>
                <td>{r.bertscore_f1.toFixed(3)}</td>
                <td>{r.completion_perplexity.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="workflow-controls">
        <label>
          Model
          <select value={model} onChange={(e) => setModel(e.target.value)}>
            {data.evaluation.runs.map((r: any) => (
              <option key={r.model}>{r.model}</option>
            ))}
          </select>
        </label>
        <label>
          Heldout case
          <select value={index} onChange={(e) => setIndex(+e.target.value)}>
            {run.rows.map((r: any, i: number) => (
              <option key={r.id} value={i}>
                {r.id}
              </option>
            ))}
          </select>
        </label>
        <button
          onClick={() => {
            setModel("lora");
            setIndex(0);
            setProbe("paraphrase");
          }}
        >
          Reset evaluation
        </button>
      </div>
      <div aria-live="polite">
        <p>
          <strong>Instruction:</strong> {row.instruction}
        </p>
        <p>
          <strong>Reference:</strong> {row.reference}
        </p>
        <p>
          <strong>Actual output:</strong> {row.prediction || "[empty]"}
        </p>
        <p>
          Category: {row.error_category}. This is an automatic diagnostic, not a
          human factuality verdict.
        </p>
      </div>
      <h4>Authored disagreement probes</h4>
      <label>
        Constructed candidate
        <select value={probe} onChange={(e) => setProbe(e.target.value)}>
          {data.evaluation.metric_probes.rows.map((r: any) => (
            <option key={r.id}>{r.id}</option>
          ))}
        </select>
      </label>
      <p>Reference: {p.reference}</p>
      <p>Candidate: {p.prediction}</p>
      <p>
        ROUGE-L {p.rougeL.toFixed(3)}; tiny BERTScore{" "}
        {p.bertscore_f1.toFixed(3)}. Compare a wrong date with a valid
        paraphrase before deciding what high similarity means.
      </p>
      <p>
        No human or hosted-judge scores have been supplied. The offline importer
        reports {data.evaluation.judgments.missing.length} missing model/case
        judgments. BERTScore uses a two-layer tiny encoder with no IDF or
        baseline rescaling; these scores are not standard RoBERTa-large results.
      </p>
    </section>
  );
}

export function QuantizedResults() {
  const [show, setShow] = useState(false);
  return (
    <section
      className="workflow-lab generation-lab"
      aria-label="Verified NF4 configuration boundary"
    >
      <p className="workflow-eyebrow">
        Actual Linux CPU quantized adaptation · separate from GPU profile
      </p>
      <h3>What did the NF4 run verify?</h3>
      <p>
        24 actual Linear4bit projections use NF4 and nested scale quantization.
        Only the 12,288 floating LoRA parameters train. Float32 compute/adapters
        and torch AdamW are explicit; paged CUDA optimization remains
        unverified.
      </p>
      <button onClick={() => setShow(!show)} aria-expanded={show}>
        {show ? "Hide" : "Show"} precision and merge checklist
      </button>
      {show && (
        <ul>
          <li>
            Frozen quantized base; real adapter gradients and selected trained
            checkpoint.
          </li>
          <li>
            Adapter saved and reloaded against a fresh identical NF4 base.
          </li>
          <li>
            Fresh quantized base dequantized before adapter attachment and
            floating merge.
          </li>
          <li>
            Logit differences bounded on a declared probe; merged artifact
            reloaded.
          </li>
          <li>
            Dequantization preserves the approximation, rather than recovering
            original checkpoint weights.
          </li>
        </ul>
      )}
      <p>
        Read the execution record for exact Linux backend, runtime, process peak
        and source/artifact hashes. These CPU checks do not establish GPU
        capacity, throughput or large-model quality.
      </p>
    </section>
  );
}
