# E2 reference workflows

Use the repository root and install [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv sync --locked --project examples/training/reference
python3 scripts/run-examples.py --id tokenizer-comparison --allow-network
python3 scripts/run-examples.py --id lora-tiny-training
python3 scripts/run-examples.py --id lora-pretrained --allow-network
python3 scripts/run-examples.py --id ner-training --allow-network
```

The public downloads need no account or API. Later replace `--allow-network` with `--offline` for cached runs. Python 3.11, PyTorch 2.8.0, Transformers 4.57.1, NumPy 2.2.6, and PEFT 0.17.0 are locked with transitive packages. Linux/Windows select CPU PyTorch wheels; this implementation was trained and measured on macOS arm64 CPU. CUDA, quantization, and large-model configurations require separate verification.

## What each run establishes

| ID | Task | Checks and limits |
| --- | --- | --- |
| tokenizer-comparison | BERT WordPiece and GPT-2 byte BPE | Two pinned vocabularies, eight English/multilingual/code cases, masks, offsets, empty input, GPT-2 round trip. Tokenizer files only; no language model weights. |
| lora-tiny-training | Known synthetic rank-one regression | Frozen base/full/adapter comparison, first-step gradients, held-out error, factorization, merge, save/reload. Target is deliberately rank-one representable. |
| lora-pretrained | Pythia-14m completion-only adapter | 40 CPU steps, validation checkpoint selection, frozen-weight digest, test loss, greedy completion, adapter reload and merging. Original templated task; not broad instruction-following evidence. |
| ner-training | Tiny BERT word-level NER | 90 CPU steps, first-subtoken labels, explicit truncation rejection, frozen embeddings, validation selection, full sentence predictions, exact typed spans, save/reload. Original controlled templates and disjoint name sets; not a real-domain benchmark. |

Model revisions are recorded in `../../manifest.json`. Pythia weights are roughly 54 MB in float32, tiny BERT roughly 18 MB, and tokenizer files add several MB. Environment installation is additional disk use. Measured runtime/memory and dependency snapshots are in `../../verification/records/`. The scripts save local artifacts under `examples/artifacts/`; these are ignored by Git.

## Inputs, outputs, and evaluation

Fixtures live under `examples/data/fixtures/`; corresponding manifests document license, hash, annotation policy, split membership, and leakage limits. NER has 26/6/8 documents, with names disjoint across train/validation/test. LoRA has 24/4/4 prompts with disjoint queried words. Shared templates make both demonstrations easier than an independent realistic dataset.

NER selects the first maximum validation-F1 checkpoint at steps 30/60/90. The all-O baseline and selected model use the same eight test documents. Invalid predicted I-tags explicitly start a repaired span; gold annotation should be valid BIO. Strict scoring compares the entire repaired typed span and retains document identity. It is not a claim of equivalence to a particular library's invalid-BIO mode.

LoRA selects minimum validation loss at steps 10/20/30/40. The frozen pretrained model and selected adapter use the same four held-out prompts. Prompt/padding labels are masked; answer and EOS labels remain. The model handles the next-token shift. Floating-point accumulation order differs after merging: the float32 path bounds global scale-relative logit drift; a second float64 check bounds maximum absolute drift below 1e-9. The original over-strict elementwise float32 check failed and its record is retained. A looser check alone would not explain the failure; the independent higher-precision comparison verifies the algebra.

## Tests and notebooks

```bash
npm run test:reference
npm run test:examples
examples/training/reference/.venv/bin/python examples/training/reference/test_model_contracts.py
```

The last command uses the actual cached tokenizer to test split words, specials, padding, truncation, and no-entity inputs. Standard-library tests check span metrics against hand-scored cases and browser/Python parity. The tiny regression checks gradients and parameter freezing during execution.

`notebooks/` contains generated source-import notebooks. Use the locked reference interpreter as your kernel and run cells from top to bottom. Their code cells are also exercised in fresh Python processes. They import the same implementation as the CLI. Interactive Jupyter tooling is optional and not in the lock.

## Failure analysis and maintenance

Do not name a local script `tokenizers.py`: it shadows the third-party `tokenizers` module and can produce circular-import errors. Those initial failures are retained in the run records. A download failure, timeout, or invalid mask must remain a failure. Repair the cause and rerun instead of editing a record into a pass.

Use `--record` to refresh execution evidence after source changes, then `npm run examples:sync`. Publication checks reject source, helper, lock, contract, or published-result drift. New helper imports must be listed in `supportFiles`. Recorded frontend data is derived from hashed run artifacts. GPU/QLoRA and broader corpora are later work; do not reuse these badges for them.
