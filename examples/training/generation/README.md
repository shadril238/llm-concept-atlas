# Generation adaptation: L21–L23

This is real optimization, not a replay of supplied labels. The website replays recorded results and calculates explicit mechanisms; the CLI/notebooks download and train tiny public pretrained models. Independent review is pending. The full E8 gate remains open for supported GPU QLoRA evidence. The Linux CPU NF4 run does not certify paged CUDA optimization or the book-scale TinyLlama recipe.

## Reproduce

From the repository root, Python 3.11:

```sh
uv sync --locked --project examples/training/generation
python3 scripts/run-examples.py --id instruction-adaptation --allow-network --record
python3 scripts/run-examples.py --id preference-adaptation --offline --record
python3 scripts/run-examples.py --id generation-evaluation --offline --record
examples/training/generation/.venv/bin/python examples/training/generation/test_model_contracts.py
python3 scripts/sync-examples.py
```

The lock pins torch2.8.0, Transformers4.57.1, PEFT0.17.1, accelerate1.12.0, BERTScore0.3.13, sacrebleu2.5.1 and rouge-score0.1.2. CPU float32, one thread, seed42. Pythia14m weights are roughly54 MB; tiny BERT18 MB; installation/caches/checkpoints add disk and RAM. `models.json` pins every downloaded file's revision, size and SHA. Nothing uses remote model Python, API keys or hosted inference. Later cached runs use `--offline`. Published snapshots preserve exact JSON artifact bytes and execution records bind source, support files, lock, dataset and contract hashes.

Run the three notebooks in order with this environment's kernel: instruction-adaptation, preference-adaptation, generation-evaluation. Each notebook has provenance, prediction, actual shared-source execution, output interpretation, transfer question and artifact reload. `verify_notebooks.py` creates a separate fresh Jupyter kernel for each notebook and writes hash-bound `notebook-execution.json`; executed notebooks stay ignored under artifacts. Run notebooks **before** final CLI `--record` commands because both overwrite artifacts. Regenerate with `python3 scripts/sync-examples.py`; do not format or edit generated notebooks independently.

## Data and licensing

`instruction-dolly.json` contains the first48 unique-context information-extraction rows with31–85 context words and1–12 answer words, split in source order32/8/8. These deterministic filters were fixed before fitting. The immutable dataset revision and upstream SHA are in its manifest. References are retained verbatim; filtering/IDs/splits are the modifications. Exact source row, prompt and context cannot cross splits. Categories recur. Pretraining overlap is unknown and the small subset is not a benchmark. An original two-row MIT inspection fixture is not mixed into evaluated data.

Dolly was authored by Databricks employees. The subset and its derived preference data remain **CC BY-SA3.0**, separate from MIT code. See the [upstream card](https://huggingface.co/datasets/databricks/databricks-dolly-15k) and [license](https://creativecommons.org/licenses/by-sa/3.0/). Preserve attribution and modification notices when redistributing derivatives. The preference fixture introduces rule-authored unsupported-abstention/irrelevant-addition/swapped-answer/unwarranted-refusal alternatives. These are pedagogical relative labels, not independent human feedback; source answers still need correctness review. Candidate lengths, ties and error slices remain visible.

Pythia is Apache2.0 and tiny BERT is MIT under their pinned source cards. Model files/trained binaries and the book PDF are not committed. Generated text is raw model output, not an endorsed factual source. No book passages, figures or artwork are copied into lessons.

## Training contracts

Pythia has no native chat template: the original ordinary-text `### Instruction/Context/Response` format is intentional. Prompt and completion are separately tokenized under an explicit boundary contract; no BOS/special duplication is implied. The real EOS is appended and supervised. Prompt/padding labels are−100; attention padding is separate. The model's causal loss shifts once. Overflow256 fails, including empty targets, rather than silently destroying supervision. Actual token traces are published.

Full:40 steps, all14,067,712 parameters, LR0.0002. LoRA:60 steps,12,288 trainable factors, six fused QKV 384×128 projections, rank 4/alpha 8/dropout0/biasnone, LR0.003. Each update averages two microbatch token-mean losses of two examples, so effective example batch is 4; this is not identical token weighting to concatenating unequal-length microbatches. AdamW, weight decay0, constant LR/no warmup, clip norm1. Validation token-weighted completion NLL every10 selects among trained checkpoints; untouched baseline is separate. Unequal method budgets are explicit.

Save/reload selected full model and adapter; verify frozen base bytes, updates and nonzero gradients. Float32 merge maximum logit error is bounded by 0.001×max(1, global logit scale) on the fixed probe, with actual value published; a float64 comparison independently checks algebra below 1e-8. This accommodates normalization-amplified floating summation differences, not arbitrary model mismatch. Reload the saved merged artifact. Identity includes exact base revision, selected adapter and tokenizer/template. This does not certify all prompts or all precisions.

Reward: tiny BERT plus newly initialized scalar head, all parameters trained,60 steps/batch 4/LR0.0003. Pairwise loss−logsigmoid(chosen_reward−rejected_reward). Validation pair loss selects trained checkpoints; baseline/heldout ranking, ties, lengths and shorter-answer baseline are recorded. Saved reward model reload reproduces scores.

DPO: fresh LoRA policy and frozen reference start from the selected merged SFT checkpoint; the separate reward model is never queried. Initial relative margins are zero and loss log2. Forty steps/batch 2/LR0.001/beta 0.1; sum completion log probabilities including EOS, prompt/pad masked, original sigmoid loss, no length normalization/smoothing/reference refresh. Validation DPO loss selects trained checkpoints. Frozen base and reference hashes remain fixed; fresh adapter reload reproduces scores. PPO training is not executed. Explicit loops follow pinned TRL's documented sigmoid objective, not its convenience Trainer API.

## Evaluation and honest limits

All four policies use identical eight heldout tasks and greedy max_new_tokens24. Preserve empty and limit-reached outputs. Completion PPL uses the same tokenizer/mask and token-weighted NLL; normalized exact match uses lowercase word tokens. Sacrebleu signature, ROUGE-L F1 and real BERTScore settings are retained. Tiny BERTScore uses two layers/no IDF/no rescaling; it is not canonical RoBERTa-large and cannot be compared across encoders. Authored paraphrase/wrong-day probes are separate from actual model outputs.

The author suite has0/8 generated exact matches for every policy. LoRA lowered completion NLL, full worsened it, and DPO worsened completion PPL despite good authored-pair preference margins. These are failure analyses, not a production extraction result or proof of general alignment. Human ratings remain missing; automatic reference mismatch is not a hallucination verdict.

## Offline human/judge workflow

No hosted client is implemented. First calibrate a rubric on independent human-adjudicated examples, including correct paraphrases, wrong facts, empty answers and ambiguous references. Record rater/judge model+revision, full prompt/rubric version and SHA, order and exact output SHA. Rubric `atlas-e8-v1`: grounding, instruction adherence and clarity, each0–2. Model identity stays hidden from raters; AB/BA means candidate/reference presentation order, not a training-label swap. Retain ties/disagreement and adjudication. No human/calibration result is fabricated here.

```sh
python3 examples/training/generation/import_judgments.py --template examples/artifacts/e8-evaluation/packet.json
python3 examples/training/generation/import_judgments.py --ratings examples/artifacts/e8-evaluation/ratings.json
python3 examples/training/generation/import_judgments.py --ratings examples/artifacts/e8-evaluation/ratings-ab.json --compare-order examples/artifacts/e8-evaluation/ratings-ba.json
```

The template leaves all ratings and judge versions blank; a separate `.mapping.json` unblinds system IDs. Use `--unblind-map path/to/packet.mapping.json` to join a completed packet to its system/id mapping, or supply normalized rows with `model` directly. Required fields: model,id,output_sha256,rubric_version,judge_version,judge_prompt_version,judge_prompt_sha256,order,grounding,instruction,clarity. Fingerprint the actual full rubric/prompt using SHA256. The importer rejects duplicates/unknowns/stale outputs/invalid scores and explicitly returns missing cases. Order comparison requires opposite order and reports matched denominators/disagreements. Do not use test ratings for subsequent checkpoint selection. Optional local `examples/data/local/generation-judgments.json` is read only when explicitly present during evaluation; local labels are not silently added to published evidence.

## Recovery and resources

Missing environment: repeat locked uv sync. Network/download error: retain pins/checksums and retry the exact public file; do not silently select a different revision. Corrupt cached weights: remove only the reported file under ignored `.cache/e8`, then redownload. Overflow: revise a declared dataset/profile and rerun provenance/tests; do not truncate away the target. Missing SFT model: run L21 before L23. A failed reload or wrong-base identity is an error, not a skipped check. Lower batch/sequence/model changes require new records; the published memory measurement is cumulative child process peak RSS, not GPU allocation.

For genuine NF4 CPU training use Linux AVX2/aarch64 supported by bitsandbytes 0.48.2, then:

```sh
uv sync --locked --project examples/training/quantized
python3 scripts/run-examples.py --id nf4-adaptation --allow-network --record
```

This isolated lock installs actual bitsandbytes. macOS/MPS fails clearly. NF4+nested metadata, float32 compute/adapters and ordinary AdamW are explicit. Frozen quantized weights remain fixed and only adapter tensors are restored for selection; packed quantization metadata cannot be round-tripped by an ordinary in-place state_dict load. Report logical parameter counts separately from packed storage elements. Dequantize a fresh NF4 base before adapter loading/merging; use the stated0.001×max(1,global logit scale) float32 bound, independently verify float64 merge algebra below 1e-8, and use1e-5 reload tolerance on the probe. The resulting float model is the quantized approximation plus update, not original checkpoint recovery or a requantized deployment claim. GPU/paged optimization and book-scale evaluation remain open.
