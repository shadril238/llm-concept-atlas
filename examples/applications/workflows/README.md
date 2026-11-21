# E5: local prompting and bounded application workflows

L09–L11 connect eighteen lessons to real local generation, exact search, inspectable memory, and bounded tools. No hosted provider, API key, or cloud tracing is used. The published small-model failures are intentional evidence, not silently repaired outputs.

## Install and run

From the repository root, install Python 3.11, [uv](https://docs.astral.sh/uv/getting-started/installation/), a C++ toolchain, and CMake. On macOS, Xcode Command Line Tools provide the compiler. Then:

```sh
uv sync --locked --project examples/applications/workflows
python3 scripts/run-examples.py --id prompt-evaluation --allow-network
python3 scripts/run-examples.py --id search-and-voting --offline
python3 scripts/run-examples.py --id local-model-chain --offline
python3 scripts/run-examples.py --id memory-policies --offline
python3 scripts/run-examples.py --id bounded-tool-workflow --offline
```

The isolated lock pins llama-cpp-python 0.3.37, langchain-core 1.8.4, NumPy 2.3.5, Pydantic 2.12.5, and Jinja2 3.1.6. Native compilation is limited to four build workers; runtime uses one CPU thread. Installation time is separate from model timings. The first public-model download is 144,811,360 bytes, plus package/build downloads. Later runs check and reuse `.cache/e5/`; `--offline` refuses a missing model cache. Model outputs and full search traces are written to ignored `examples/artifacts/e5-*/`. Do not commit model files or the source PDF.

The author executed all five sources and all five notebooks with **fresh actual Jupyter kernels and every code cell in order** on macOS/Python 3.11.18. The generated notebooks include setup, fixture inspection, predictions, actual shared-source execution, result/failure inspection, saved-result reload, and transfer exercises. They import the verified modules rather than copying the model algorithm. Pinned ipykernel 7.1.0, nbclient 0.10.4, and nbformat 5.10.4 are included in this environment. Select this environment's Python kernel in your notebook editor and start in this project or its `notebooks/` directory.

Reproduce Restart → Run All without changing user kernel registrations:

```sh
examples/applications/workflows/.venv/bin/python examples/applications/workflows/verify_notebooks.py
```

Use `--allow-network` for the initial public-model download if needed. Each notebook gets a new kernel; task-local Jupyter/IPython directories avoid editing user settings. The checked source/lock/notebook hashes are in `notebook-execution.json`; executed copies with outputs stay in ignored `examples/artifacts/e5-notebooks/`. This verifies kernel execution, not an independent notebook-editor usability study. Run notebooks **before** final CLI `--record` commands because they overwrite timing-bearing artifacts.

## Exact artifact and runtime contract

- Quantized artifact: `bartowski/SmolLM2-135M-Instruct-GGUF` at `09816acd5d99df7be770d85ea30822623dab342c`, filename `SmolLM2-135M-Instruct-Q8_0.gguf`.
- SHA-256: `5a1395716f7913741cc51d98581b9b1228d80987a9f7d3664106742eb06bba83`.
- Original author/model: [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct), whose model card declares Apache 2.0. The pinned quantization repository is a separate artifact; no exact upstream precision-equivalence experiment is claimed.
- Actual metadata: llama architecture, `general.file_type=7` (mostly Q8_0), embedded ChatML template, training context 8192. Runtime context is deliberately 768, batch 256, one thread, zero GPU layers, `offload_kqv=False`, `op_offload=False`.
- Renderer equivalence is checked against the embedded Jinja template using fixed typed messages. The controlled renderer rejects literal ChatML role delimiters inside message content. This is not a general untrusted-input isolation guarantee.
- Native JSON constraints pass `LlamaGrammar.from_json_schema` to `create_completion`; no post-generation repair is used. A one-token cap probe demonstrates that truncation can still leave incomplete JSON.

On the author's macOS build, llama-cpp-python 0.3.37 forces Metal backend initialization even with CPU runtime placement. A managed sandbox denied its command queue and failed to create a context. Running the same source from the normal host succeeded; native diagnostics confirmed CPU KV and compute buffers, with zero Metal compute allocation. This is CPU computation, not a claim of a Metal-free compiled binary. A sandbox initialization error is distinct from a bad GGUF checksum or insufficient context. Linux paths use the same pinned sources but have not been independently executed by the author; the optional CI job can provide a separate Linux execution record.

## Read the results honestly

L09 freezes six arithmetic and three extraction variants before the recorded evaluation. Demonstrations use only the original design split. Each task records the full prompt, examples, native token counts, output cap, finish reason, raw output, extraction result, and reference comparison. The parser is deliberately strict: zero extracted-answer accuracy does not mean no human could infer an answer from any prose. Requested format and reference correctness are different metrics. Nine valid constrained order objects include only two fully correct objects in the author run. Single sequential wall timings reuse a loaded runtime and prefix cache; they are not a speed leaderboard, monetary cost, or peak-memory measurement.

L10 samples five outputs per case/condition. Its free-format and native-integer paths change both instruction and decoding, so they are separately labeled mechanism probes. A unique valid plurality wins; ties and empty valid sets abstain. Strict majority uses every sample, including invalids. Exact rational Game-24 search compares complete/beam/budget policies, saves full traces, and checks AST operators and original input leaves. The model-guided variant only ranks up to eight pre-generated legal options, at most three calls. Failure under pruning or a work cap is not proof of impossibility.

L11a verifies current LangChain typed-message/Runnable adapters against plain Python, then performs an actual two-model-call chain with intermediate extraction and deterministic pricing. Extraction, copy consistency, and end-to-end correctness are separate checks. L11b compares buffer/window/generated-summary policies on the identical eight-message conversation with 192/384-token application budgets including a 64-token output reserve. Summary updates are actual model calls; source IDs, policy omissions, budget evictions, and literal evidence diagnostics are recorded. The window means two complete user/assistant pairs, not two messages.

L11c runs finite local tools through plain Python and actual LangChain StructuredTool paths. Native model choice IDs resolve to predeclared operands/actions; the model does not freely extract arbitrary tool arguments. A tool observation is required before final output, observations are capped at 160 characters, and the loop stops at three steps. Scripted/FakeListLLM tests separately force invalid arguments, recovery, and max-step termination; these do not establish model competence. Actual model cases preserve poor choices and report only two correct results out of six task/path cases in the author run.

## Reproduce and verify changes

Run offline behavioral contracts with `python3 -m unittest discover -s tests/examples -v` and `npm run test:workflows`. These cover duplicate JSON keys, strict numeric types, extraction boundaries, plurality/majority/ties, forbidden ASTs, exact fractional solutions, pruning versus budget stops, tool argument rejection, bounded termination, whole-turn packing, and browser/Python agreement. All five real model scripts assert runtime/contract invariants without requiring an invented quality score.

After editing a model source or its support files, rerun the relevant example with `--record`. Execute notebook cells **before** creating final records: notebook runs overwrite artifact timing fields. Then run `python3 scripts/sync-examples.py`, reassess lesson/audit hashes, and run `npm run release:build`. Do not edit a verification record to make a changed artifact appear verified. Author execution records are distinct from independent technical/editorial acceptance and beginner feedback, which remain pending.

The adaptation follows Chapters 6–7 and the pinned official notebooks. It replaces book-era LLMChain/memory/agent imports with explicit langchain_core Runnable, typed-message, and StructuredTool APIs. The book's FP16 filename does not establish an eight-bit artifact; GGUF is a container, and this run verifies its actual Q8 type. Original tasks, diagrams, examples, and prose are MIT atlas work. No copied book pages or artwork are published.
