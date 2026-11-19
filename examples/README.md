# Reproducible atlas examples

Run commands from the repository root. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first. E1 was verified with uv 0.13.9 and Python 3.11.18 on macOS arm64. Other platforms use the lock's platform wheels but are not yet execution-verified.

```bash
uv sync --locked --project examples/foundations
uv sync --locked --project examples/applications
npm run examples:list
npm run check:example-manifest
python3 scripts/run-examples.py                     # offline foundations only
python3 scripts/run-examples.py --id embedding-retrieval --allow-network
python3 scripts/run-examples.py --id embedding-retrieval --offline
```

The registry contains the original 98 Python blocks, two E1 examples, and six E2 reference examples, and 38 E3 workflow/mechanism entries. Superseded LoRA/NER/tokenizer snippets remain as historical downloadable sources; the three reference lessons now use replacements. The 76 published exact-output checks are retained. Another 19 earlier standard-library blocks now receive execution checks. These earlier examples have no new semantic correctness certification: the E0 findings still apply. Three historical unpinned model sketches remain explicitly unverified and cannot be run through this runner until migrated.

## Source authority and synchronization

`manifest.json` declares source files, named regions, environments, data, validations, and evidence. `schemas/` describes the public data contracts. `scripts/example_tools.py` enforces repository-specific constraints, including source mappings, missing registrations, notebook/lesson synchronization, and stale evidence. Run `npm run examples:sync` after changing source or recorded results; `npm run check:example-manifest` rejects drift.

The initial 98 lesson blocks have byte-equivalent Python files in `legacy/`. Existing lesson text remains under audit; edits must update both source and lesson deliberately. New lesson blocks are generated from named regions in reusable Python files. Generated notebooks import those same modules. Run notebook cells in order from the environment directory or its `notebooks/` directory, using that environment's Python kernel. The first two notebook code sequences are also exercised in fresh Python processes; interactive Jupyter frontends are optional tooling and are not included in these locks.

## Execution and evidence

```bash
python3 scripts/run-examples.py --id attention-softmax --record
python3 scripts/run-examples.py --id embedding-retrieval --offline --record
npm run examples:sync
npm run test:examples
npm run release:build
```

Records retain command, UTC timestamp, base Git commit, dirty-tree indicator, source/contract/lock/fixture hashes, model revision, installed package versions, CPU/platform, stdout/stderr, elapsed duration, peak child memory, and outcome. A dirty-tree run is identified by file hashes; the base commit alone does not identify its code. Source, validation-contract, or lock changes invalidate the evidence until rerun. Failed records remain failures and do not replace successful evidence. Never edit a failed record into a pass.

Memory is `RUSAGE_CHILDREN`'s cumulative peak within one runner invocation, including environment probes; use one `--id` per invocation for resource comparison. Download timing can dominate a first model run. Timings and memory are observations on the recorded host, not promises. Package installation disk use is separate from model weights. E1 verified CPU inference; E2 adds the [reference training environment](./training/reference/README.md). No GPU or hosted API configuration has been verified.

`--offline` disables Hugging Face network access and requires cached files. Offline standard-library examples perform no downloads. This runner executes trusted repository code locally; it is not an operating-system network sandbox. Paid APIs are never invoked by these examples or site visits.

## Data, output, and failure handling

`data/manifests/` records provenance, hashes, and split policy. The original five-sentence retrieval fixture is demonstration-only and is never used for fitting. Its two ranking checks are smoke invariants, not an accuracy estimate. The model script checks shapes, finite values, unit norms, and saving/reloading `examples/artifacts/retrieval/embeddings.npy`. Generated binary artifacts and local environments are ignored by Git.

- Missing environment: run the corresponding `uv sync --locked` command.
- Changed environment: `uv sync --locked --check --project examples/applications` detects drift; resync before verification.
- Download or certificate failure: repair network/certificate configuration and retry with `--allow-network`. No verification is awarded on failure.
- Missing cache offline: do one explicit online run first. `ATLAS_MODEL_PATH` may point to a local snapshot of the exact declared revision; record its provenance before sharing results.
- Unsupported wheel/platform: retain the failure and document another supported environment; don't silently modify the lock or claim equivalence.
- Timeout: inspect stderr and the profile. Increase a timeout only with a reason; a slower run is still subject to output checks.

## CI boundaries

Ordinary Pages builds validate manifests, synchronization, coverage, all offline calculations, and site links. They install only the foundation environment. `.github/workflows/examples.yml` separately checks foundations on pull requests; a manual `model` input enables the CPU model job. No secrets are required or exposed. Larger application/training/GPU/API environments will be added by the phases that implement their real workflows. They are not placeholders presented as verified configurations.

E3 foundations use the same locked reference environment. See [E3.md](./training/reference/E3.md) for L01–L06 commands, pretrained-vector provenance, optional larger-data limits, and CPU cache measurement details.
