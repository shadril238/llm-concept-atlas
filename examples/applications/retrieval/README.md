# E6: retrieval, grounded answers and actual multimodal models

L12–L14 connect 23 detailed Chapter 8–9 lessons to five actual model workflows and staged notebook views. Each lesson includes definition, theory, worked example, experiment, implementation, applications, limitations and topic-specific exercises. Interactive components explore saved runs or explicitly recalculate a formula; they do not run inference in the browser.

## Install and choose a runtime

Use Python 3.11 and uv. From the repository root:

```sh
uv sync --locked --project examples/applications/retrieval
python3 scripts/run-examples.py --id retrieval-benchmark --allow-network
python3 scripts/run-examples.py --id grounded-retrieval --allow-network
python3 scripts/run-examples.py --id openclip-retrieval --allow-network
python3 scripts/run-examples.py --id multimodal-chat --allow-network
```

After the first download, use `--offline` instead of `--allow-network`. The isolated lock pins PyTorch 2.8.0, torchvision 0.23.0, Transformers 4.57.1, NumPy 2.3.5, sentence-transformers 5.1.2, scikit-learn 1.7.2, open_clip_torch 3.3.0, Pillow 11.3.0 and Accelerate 1.12.0. Linux resolves the official PyTorch CPU wheel index; macOS uses its platform wheels. No API, cloud tracing, credentials or remote model code is used.

`models.json` pins exact public revisions, filenames, sizes and LFS SHA-256. The downloader validates the required files and hashes even in offline mode. Weights stay in ignored `.cache/e6/hub/`; artifact outputs stay in ignored `examples/artifacts/e6-*/`. Model storage is separate from package downloads, tensor buffers and measured process memory.

| Workflow | Actual model | Weight download, approximately | Verified profile |
| --- | --- | --- | --- |
| Retrieval | MiniLM-L6 sentence encoder + MS MARCO MiniLM cross-encoder | 182 MB combined | macOS CPU, float32, one thread |
| Grounded generation | Above + FLAN-T5-small | Additional 308 MB | macOS CPU, float32, one thread |
| Image/text search | OpenCLIP ViT-B/32 OpenAI weights | 605 MB | macOS CPU, actual open_clip library |
| Caption/VQA | Genuine Salesforce BLIP-2 OPT-2.7B | 14.98 GB, two shards | Heavy Linux CPU BF16/autocast; Q-Former components retained float32 |
| Visual conversation | SmolVLM-256M-Instruct | 513 MB | macOS CPU, float32; two preprocessing conditions |

The genuine BLIP-2 script is **Linux-only** and checks at least 10 GiB currently available memory before loading. That guard is not a sufficient total-RAM recommendation: the actual CI record measured roughly 14.4 GiB peak child memory, including environment probes. Allow comfortable headroom. It runs all weights on CPU, BF16 autocast with cache disabled, float32 image input, int64 text IDs, and 12 generated tokens per case. The model's float32 Q-Former/query components are retained by Transformers. This mixed dtype profile is explicit; it is not a fully BF16 parameter claim.

```sh
# Run only on an appropriately provisioned Linux host.
python3 scripts/run-examples.py --id blip2-captioning --allow-network
```

The successful fresh-kernel + CLI profile is [CI run 45237373909](https://github.com/shadril238/llm-concept-atlas/actions/runs/45237373909). No CUDA, four-bit or disk-offload success is claimed. An earlier float32 disk-offload attempt failed with a one-dimensional Linear weight; its failure led to this separately executed resident Linux profile. Do not treat a downloaded checkpoint or configured device map as successful inference. A 16 GB laptop with little available RAM is unsuitable for this verified heavy setup.

## Run notebooks from fresh kernels

Five generated notebooks import the same modules as the CLI; they do not duplicate model algorithms. Select this environment's Python 3.11 kernel, start in this project or its notebooks subdirectory, and Restart → Run All. Each has provenance, prediction, actual execution, output/failure inspection, saved-result reload and transfer guidance. The author executed all code cells in fresh actual Jupyter kernels, four locally and one on heavy Linux CI.

```sh
examples/applications/retrieval/.venv/bin/python examples/applications/retrieval/verify_notebooks.py
# Separate heavy Linux execution; first download may need --allow-network.
examples/applications/retrieval/.venv/bin/python examples/applications/retrieval/verify_notebooks.py --id blip2-captioning --allow-network
```

The default deliberately excludes the heavy model. Use repeated `--id` to select smaller notebooks. Task-local Jupyter paths avoid modifying user kernel registrations. `notebook-execution-local.json` and `notebook-execution-blip2.json` contain notebook/lock/verifier hashes, host, UTC time, cell count and method. Executed copies remain ignored. This is actual kernel execution, not an independent editor-usability study. Run notebooks **before** final CLI `--record` because both overwrite timing-bearing artifacts.

## L12: ranking and design/evaluation separation

The original MIT corpus has 24 fictional documents and 22 queries: six design (four answerable/two absent), sixteen evaluation (fourteen answerable/two absent). Stable source IDs, sentence/word chunk policies, zero-based Unicode code-point offsets and complete graded judgments remain inspectable. Grade 2 means direct relevance, 1 related/partial and 0 irrelevant; full-answer support uses stricter source IDs. Overlapping chunk hits collapse to unique documents before metrics.

BM25 uses a transparent positive-IDF formula. Dense encoding uses actual MiniLM masked mean pooling, 384-dimensional unit vectors and an sklearn brute-force cosine index, cross-checked against independent Torch dot products and after save/reload. The reranker scores only six dense candidate chunks; it cannot recover outside that pool. P@k divides by k even with an unfilled list. AP@k divides by **all relevant documents**, not min(R,k). MAP excludes undefined absent-query AP and reports fourteen answerable queries out of sixteen. nDCG uses gain 2^grade−1. Absent queries are retained for false-answer/threshold analysis. The six design queries alone select cosine threshold 0.4 from declared candidates.

Recorded MAP@3 is about 0.768 lexical, 0.933 dense and 0.952 reranked. These are small author-judged results, not a broad leaderboard. Cross-encoder logits, BM25 and cosine have distinct scales; none is calibrated answer probability. Changed word chunks are a separately recomputed condition. Original evaluation references are never model prompt inputs.

## L13: genuine generation and semantic review

Eight inspected cases × six strategies produce 48 actual drafts. Rewriting/multi-query/dependent multi-hop use actual generated strings and prior retrieved evidence. Routing uses fixed design examples with an explicit invalid-route fallback. The agent scores a predeclared finite collection domain with actual FLAN-T5 mean target-token likelihood, requires evidence before finish, and stops on three searches or repeated sources. It is not unrestricted planning, native grammar generation or arbitrary argument extraction. Scored tokens are separate from generated tokens.

Contexts contain complete deduplicated passages and source IDs under a 384-input-token application budget. The threshold can withhold context and override the displayed answer with UNKNOWN; the raw model call is still retained. Model abstention and controller withholding differ. Imported source instructions remain source text, not commands.

The author inspected all raw drafts and supplied hash-bound proposition judgments in `rag-author-review.json`. Only 9/48 answer the exact question correctly in this run. Correctness, supplied-context faithfulness, fluency (0–3) and usefulness (0–2) have explicit rubrics and rationales. `substring_probe` is a labeled diagnostic, not semantic correctness. A true number can refer to the wrong entity. Correct but uncited answers, faithful irrelevant statements, unsupported values and empty/bare-ID outputs remain visible. Uncertain judgments stay null.

Citation precision grades each link's useful support for its associated proposition. Citation recall grades full support by the **joint cited set**; one partial useful link is insufficient. No links gives undefined precision, no factual propositions gives undefined recall. Source-ID resolution and exact spans do not establish semantic entailment. Author review is not independent acceptance. Because these L13 probes were inspected during implementation, do not call them untouched generation evaluation; freeze new cases for a generalization claim.

## L14: actual images, patches, bridge and conversation

The three source photos retain original PNG bytes and per-file attribution in `vision-images.json`: NASA public domain, Stefan van der Walt CC0 and Rachel Michetti CC0. They are familiar images with unknown training overlap, not an independent benchmark. Separate processed previews inverse-normalize actual tensor pixels and round for display; image/pixel hashes, dimensions, crop box, retained area, normalization and text IDs remain recorded.

The actual open_clip library loads frozen ViT-B/32 weights. Manual preprocessing matches native preprocessing. Unfolding 49 actual patches and multiplying learned projection weights matches the native convolution; class/position shapes are recorded. Actual image/text unit vectors yield retrieval matrices, changed wording conditions, and a symmetric paired loss checked against PyTorch cross entropy. Stored embeddings reload to the same matrix. No CLIP training is performed. Candidate-set softmax is relative, not calibrated object presence.

Genuine BLIP-2 forward hooks record [1,257,1408] vision tokens → [1,32,768] learned-query states → [1,32,2560] language projection. This is inference with a historically trained bridge, not bridge training. Caption output mentions a shuttle; judge that detail against the portrait. Coffee VQA says Espresso, which is more specific than the image can conclusively establish. Raw decoded BLIP output retains the question prompt; IDs expose the boundary.

SmolVLM is a **separate** 256M model, not a substitute for BLIP-2. Both single-image and split-image settings run two real turns, with the first actual generated answer in the second prompt. The single-image price response abstains, while split-image response invents “100” without a visible price. First-turn preprocessing is isolated; second-turn comparison also carries changed generated history and is explicitly a mediated comparison. Continuation-only decoding is checked. Neither fluent wording nor more image crops establishes support.

## Optional offline imports: zero provider calls

Keep hosted comparisons opt-in and external. Freeze the same sources, questions, context policy and prompt version before collecting new outputs. Save every requested ID, error, raw answer and citation span. Export JSON with this shape (illustrative fields, not executed provider output):

```json
{
  "provider": "your-provider",
  "model": "exact-model-version",
  "prompt_version": "your-frozen-v1",
  "collected_at": "ISO-8601-time",
  "outputs": [
    {"id": "eval-hours", "answer": "raw answer", "citations": []},
    {"id": "eval-pool", "answer": null, "error": "request failure"}
  ]
}
```

Each optional citation has `source_id`, `answer_start`, `answer_end`, `source_start`, `source_end`, `quote`. Offsets are exact zero-based Unicode code points, end-exclusive, not UTF-8 bytes or JavaScript UTF-16 units. The quote must equal the source slice. Unknown/duplicate IDs, design IDs, booleans as offsets, unresolved sources and mismatched quotes are rejected. Missing requests/errors remain visible. No semantic support is inferred from span validity.

```sh
python3 examples/applications/retrieval/import_external.py grounded /path/to/export.json
python3 examples/applications/retrieval/import_external.py judge /path/to/judgments.json
```

A judge envelope has `judge_model`, `rubric_version`, and `rows`. Each row has `id`, `variant`, `raw_sha256`, `support` (`supported`, `unsupported`, `uncertain`) and integer `fluency` 1–3. Compute the SHA-256 of the exact raw draft's UTF-8 bytes. Unknown/duplicate cases, edited drafts and invalid labels fail validation. These externally supplied labels are not independent truth: compare against the human rubric, publish disagreements and uncertainty, and retain judge provenance. Shared biases, style preference, source-instruction susceptibility and prompt variance limit a judge. The importer makes zero API requests; hosted inference and model judging remain unverified.

## Verification and changes

Run `python3 -m unittest discover -s tests/examples -v`, `npm run test:retrieval` and `npm run release:build`. Tests cover denominator/absence behavior, duplicates, graded relevance, Unicode chunks/spans, BM25 empty documents, multi-source citation support, external import errors, fresh-kernel evidence, image hashes and real-history continuity. JS/Python calculations agree. Model scripts assert execution contracts without imposing an invented quality target.

After changing a model source, lock, fixture or support dependency, rerun its actual notebook and CLI `--record`. BLIP-2 evidence must be regenerated on a suitable Linux host; editing a record is not verification. After final records run `python3 scripts/sync-examples.py`, rerun affected offline evidence examples with `--record`, update author audits, and run the release gate. Keep model weights and the source PDF ignored.

Original explanations/code follow Chapters 8–9 and the pinned authors' notebook snapshot ea3390819997999a51983677b80b3aac4dc50ada. Book prose/code sometimes name different embedding models; this adaptation explicitly uses MiniLM. The book's OpenCLIP discussion uses Transformers CLIPModel; this implementation uses the actual open_clip package. Text EOS pooling is distinct from image class-token pooling. Retrieval encoder training/adaptation belongs to E7; inference/indexing alone does not train it. No book artwork or pages are redistributed. Independent technical/editorial review and beginner feedback remain pending.
