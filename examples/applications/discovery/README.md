# E4: evaluated classification and topic discovery

This isolated CPU environment implements Chapters 4–5/L07–L08. It builds on E1's pinned encoder approach without changing E1/E2/E3 locks or execution evidence. Python 3.11, PyTorch 2.8.0, Transformers 4.57.1, NumPy 2.3.5, scikit-learn 1.7.2, sentence-transformers 5.1.2, BERTopic 0.17.4, UMAP 0.5.9.post2, HDBSCAN 0.8.40, and sentencepiece 0.2.1 are direct pins. All transitives/hashes are in `uv.lock`. Later BERTopic versions have different APIs/serialization behavior; these examples target the declared lock.

## Run

From repository root:

```bash
uv sync --locked --project examples/applications/discovery
python3 scripts/run-examples.py --id classification-comparison --allow-network --record
python3 scripts/run-examples.py --id modular-bertopic --allow-network --record
python3 scripts/run-examples.py --id topic-label-generation --allow-network --record
npm run examples:sync
```

Use `--offline` after caches are populated. No API account, paid request, or GPU is needed. The first two runs are independent; local topic labels require the fitted L08 artifact. Run these recording commands sequentially because they update a shared manifest. The generated import-source notebooks run the same source and work from this environment directory or its `notebooks/` child.

The extra `bertopic-pipeline` registry entry extracts the actual pipeline function into the concept lesson and executes the same full workflow. If running it, run it **before** the final `modular-bertopic` recording: both write the same local artifact, and the latter supplies the published artifact hash. This explicit dependency prevents a later unrecorded overwrite from being treated as the published result.

Direct dependencies take hundreds of MB; initial public model weights total roughly 1 GB in addition to installed packages. First download and JIT compilation can be much slower than warmed execution. Reference records report actual total process time and conservative child-process peak memory; method timings separately exclude loading and warm-up.

## Pinned public models

| Role | Model | Revision | Model-card license |
| --- | --- | --- | --- |
| Frozen document/sentence vectors | `sentence-transformers/paraphrase-MiniLM-L3-v2` | `4ca70771034acceecb2e72475f72050fcdde4ddc` | Apache-2.0 |
| Task head | `distilbert/distilbert-base-uncased-finetuned-sst-2-english` | `714eb0fa89d2f80546fda750413ed43d93601a13` | Apache-2.0 |
| Entailment classifier | `cross-encoder/nli-MiniLM2-L6-H768` | `b95119ce93d3e065de6214e38cd4a97b0f2f2c6d` | Apache-2.0 |
| Label generation | `google/flan-t5-small` | `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab` | Apache-2.0 |

Public cards: [MiniLM](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2), [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english), [NLI](https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768), [FLAN-T5](https://huggingface.co/google/flan-t5-small). No remote custom model code is executed.

## L07 data provenance and task policy

**Attribution:** Kotzias, D. (2015). *Sentiment Labelled Sentences*. UCI Machine Learning Repository. [DOI 10.24432/C57604](https://doi.org/10.24432/C57604). Source creators include Dimitrios Kotzias, Misha Denil, Nando de Freitas, and Padhraic Smyth. See their *From Group to Individual Labels Using Deep Features*, KDD 2015.

The IMDB sentence source is distributed by UCI under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Those rows remain under that license with attribution. Repository MIT applies to original code and original challenge probes, not to this externally attributed text. The source aims at positive/negative sentences rather than neutral inputs. Corpus language and annotation policy limit generalization.

- [UCI source page](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences).
- Archive URL: `https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip`.
- Archive SHA-256: `afc26626d710899948693e1a61405dce197f57ffa719fa1130d346b4cc095343`.
- `imdb_labelled.txt` SHA-256: `aef2e49e3da25714d61175e3a6e68eeef74a20a2f914318dc3be9947ea86512d`.
- Preparation source: `prepare_sentiment.py`; it reads only the declared ZIP member, never extracts arbitrary archive paths, and checks both hashes.
- Two physical-line continuations are joined with spaces before parsing their final tab/label. Canonical exact duplicates are removed globally. Canonicalization lowercases and replaces non-word runs with spaces; it is a deduplication key, not model preprocessing.
- Sort each label group by SHA-256 of `atlas-e4-v1:` plus the stable physical-line source ID. Select 100/class training, 20/class validation, and 40/class test. Do not change this subset after looking at model performance.
- Add 12 separately identified original probes. Mixed/ambiguous probe gold labels are author judgments and are not merged into the UCI headline score.

The versioned fixture and manifest are in `examples/data/fixtures/sentiment.json` and `examples/data/manifests/sentiment.json`. `python examples/applications/discovery/prepare_sentiment.py` rebuilds the subset from the source archive. Network access is explicit in that preparation command; the normal model runner consumes the committed fixture and does not download data.

No source-review/reviewer IDs are available, so exact deduplication does not prove group-level independence. Prior pretraining/instruction-tuning overlap with this review source is unknown. Do not call this a contamination-free benchmark.

## L07 computation, selection, and evaluation

`src/atlas_discovery/classification.py` fits a train-only TF-IDF/logistic baseline and a logistic head over actual frozen, mask-aware, normalized 384D vectors. Both choose C from `[0.1,1,10]` using validation macro-F1, with a smaller-C tie break. Class-average prototypes use training labels; label-description vectors do not. The latter also have a predeclared alternate-wording pass whose flips are reported, not used to choose a test winner.

Task-head probabilities are mapped through the actual checkpoint's `config.id2label`. NLI selects the declared entailment column and softmaxes those logits across two mutually exclusive label hypotheses. This is relative candidate scoring, not per-pair three-way NLI probability or calibrated confidence. FLAN-T5 uses a fixed prompt, greedy decoding, and at most eight new tokens. All transformer inputs cap at 256 tokens, including prompts/hypotheses. Similarity methods do not expose a fabricated probability.

All seven methods predict the same 80 test IDs and the same 12 separate challenge IDs. Metrics include accuracy, macro-F1 over the fixed labels 0/1, weighted-F1 by support, per-class precision/recall/F1/support, a 2×3 actual/predicted confusion matrix with an invalid column, and invalid-output rate. Zero denominators use metric 0, while accepted accuracy with no accepted cases is `null`/undefined. Invalid predictions count as false negatives for their actual class and errors in full accuracy. A handwritten small fixture checks this arithmetic independently.

The embedding-head review threshold maximizes validation coverage subject to >=90% validation accuracy. Report whether that constraint is met. A small validation set cannot guarantee production accuracy. The browser threshold control is an exploration of recorded cases and must not be used to tune on final test outcomes.

Method timings use CPU, one PyTorch thread, batch 16, a separate two-case warm-up, and three full passes over 92 rows. They include tokenization/feature extraction and exclude loading and local fitting. TF-IDF/head fitting and embedding/head fitting are recorded separately; prototype/description construction and model load times are not a complete end-to-end operational benchmark. No GPU throughput, price, or broad architecture ranking is claimed.

Artifacts: `examples/artifacts/e4-classification/results.json` and `classifier.json`. The JSON head preserves classes, weights, intercept, C, input cap, exact encoder revision, and review threshold. Reloaded sigmoid probabilities must match the fitted head to 1e-10. `src/atlas_discovery/predict.py "your review"` runs the saved head and emits label, score, and review flag without retraining. A same-width different encoder is incompatible.

## L08 corpus and real fitted pipeline

`research-summaries.json` has 45 original fictional research-style summaries (MIT), including three broad themes, cross-task bridges, and unrelated material. No theme labels, actual papers, or scientific findings are claimed. All rows participate in exploratory fitting; there is no held-out topic-accuracy claim. The corpus manifest records its exact hash and limitations.

`src/atlas_discovery/topics.py` computes embeddings, fits five-dimensional UMAP (8 neighbors, cosine input, min_dist 0, seed 42), fits HDBSCAN (minimum cluster size 5, min_samples 8, Euclidean clustering space, excess-of-mass selection), and calculates c-TF-IDF after English stop-word removal. The display independently fits two-dimensional UMAP with min_dist 0.1. A consistent seed/order helps reproducibility; different devices/versions can still affect boundaries and IDs.

The baseline has three non-noise topics and one noise point in the recorded author run. The displayed two-dimensional geometry is not the clustering geometry. Topic IDs are local identifiers. Membership strength is not probability of semantic correctness. Deliberately retained unrelated documents can be absorbed into plausible groups; inspect minority documents as well as representatives.

The default c-TF-IDF arithmetic is `n[c,t]/L[c] * ln(1 + A/f[t])`, with `A=int(mean class length)` and `f[t]` the **total token frequency** across classes, including the noise class when present. BM25-inspired and frequent-word options are disabled. A two-class worked table is checked against the actual transformer to 1e-12. Its fixed labels teach arithmetic only; no fixed labels enter the real clustering fit.

Controlled representation passes run real `KeyBERTInspired` and real `MaximalMarginalRelevance` at diversity 0.2 and 0.8. They assert unchanged assignments. KeyBERT weights are semantic similarities; MMR-selected words retain c-TF-IDF weights. High diversity can surface irrelevant group members rather than improve a label.

The density experiment changes only min_samples 8→3 on exactly the same five-dimensional points. Report the contingency table, noise transitions, and ARI, explicitly treating noise as a group in that summary. Do not compare raw integer ID equality. The heatmap uses cosine between baseline fitted topic vectors; the hierarchy uses average linkage on their cosine distances and excludes −1. Both are descriptive displays, not re-clustering or accuracy measures.

## Serialization and local generation

L08 saves topic data and c-TF-IDF using safetensors, then reloads with an explicitly supplied pinned embedding backend. It checks stored assignments/terms/weights. This lightweight format omits fitted UMAP/HDBSCAN estimators: `.transform` uses cosine to topic vectors. The lab checks that reference calculation and reports agreement with original training assignments instead of asserting a false equivalence. Refit under the declared lock/settings if you need the original fitted pipeline; do not load untrusted estimator pickles.

`topic_labels.py` loads the fitted artifact, checks unchanged assignments after an actual BERTopic `TextGeneration` update, and records its parent result-artifact hash. It uses FLAN-T5-small, an original bounded prompt, two representative excerpts of at most 35 whitespace words, and 24 greedy new tokens. The complete prompts and unedited outputs are published. A generated name is not a probability, model-weight training, factual claim, or clustering improvement.

Author review finds that the noise-group name invents a focus, Filtering underrepresents the translation group, Summarization hides unrelated members, and Acoustic speech recognition fits inspected representatives but still needs full-group review. The UI and lab keep these weaknesses visible. Independent label review remains pending.

## Optional hosted comparison

`notebooks/hosted-comparison.ipynb` is a documented workflow plus **offline importer**, not an API client. It makes zero network calls and needs no key. The notebook describes freezing shared test IDs, label schema, prompt/model version, output cap, and a bounded call/retry count before a separately authorized provider run. No hosted results are published or claimed here.

`examples/foundations/src/atlas_foundations/e4/hosted_results.py` evaluates externally collected envelopes with provider/model/prompt/date metadata. Unknown/duplicate IDs fail; missing and provider-error outputs become invalid and remain in the 80-case denominator. A JSON response object must contain exactly `label`. Its report marks provider execution as externally collected/unverified. Importer tests use explicitly synthetic results and do not count as a live API run.

Record latency, token usage, and cost only when measured. Keep credentials outside source control and the static site. Check the provider's current structured-output documentation before writing a live client; the atlas does not silently reuse credentials.

## Verification and troubleshooting

- `npm run release:build`: standard tests, audit integrity, recorded-artifact/source checks, offline mechanisms, type checks, static build, and links.
- `python3 -m unittest discover -s tests/examples -v`: split uniqueness/balance, metric boundaries/invalids, threshold coverage, c-TF-IDF quantities, MMR endpoints, and hosted-import failure handling.
- `npm run test:discovery`: browser/Python numerical agreement within floating-point tolerance.
- `examples/applications/discovery/.venv/bin/python examples/applications/discovery/test_models.py`: independent real-library c-TF-IDF, BLAS/reference, saved-head, and topic reload contracts after model artifacts exist.
- Optional `e4` workflow dispatch runs the model workflows sequentially and uploads records/results. Ordinary PR/Pages CI stays dependency-light and does not download models.
- Missing environment: install with `uv sync --locked`, not a floating `pip install` list. Cold offline cache: rerun with `--allow-network` for the public download.
- NumPy 2.2.6 initially emitted macOS matrix-operation warnings. This environment pins 2.3.5; probability/unit-norm/finite/reference checks remain active rather than suppressing warnings. See the [NumPy issue](https://github.com/numpy/numpy/issues/28687) for the related platform issue.
- BERTopic warns when saving without a model pointer and when passing an explicit assignment list during update. The lab supplies the pinned backend explicitly on load, does not perform later topic reduction, and records the updated-topic-vector distinction. These warnings describe declared choices; do not remove them by pretending a custom backend can be reconstructed automatically.
- If you change a corpus, stage, model, dtype, or lock, rerun and inspect the results before refreshing records. Execution is not independent teaching acceptance.

Primary implementation sources: [BERTopic v0.17.4](https://github.com/MaartenGr/BERTopic/tree/v0.17.4), [representation modules](https://maartengr.github.io/BERTopic/getting_started/representation/representation.html), [serialization](https://maartengr.github.io/BERTopic/getting_started/serialization/serialization.html), [UMAP parameters](https://umap-learn.readthedocs.io/en/latest/parameters.html), [HDBSCAN parameters](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html), [scikit-learn classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics). Book workflow snapshot: [Chapter 4](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/tree/ea3390819997999a51983677b80b3aac4dc50ada/chapter04), [Chapter 5](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/tree/ea3390819997999a51983677b80b3aac4dc50ada/chapter05).
