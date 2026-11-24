# E7: actual embedding and representation training

L15–L20 connect fifteen detailed Chapter 10–11 lessons to real CPU optimization, six staged notebooks, interactive evidence, task-matched evaluation and saved/reloaded artifacts. A decreasing training loss is never presented as sufficient task success. Browser controls select recorded runs or recalculate an explicitly labeled numerical/mask exercise; they do not train a model or call a provider.

## Install and run

From the repository root, use Python 3.11 and uv:

```sh
uv sync --locked --project examples/training/representation
python3 scripts/run-examples.py --id embedding-training --allow-network
python3 scripts/run-examples.py --id augmented-sbert-training --offline
python3 scripts/run-examples.py --id tsdae-adaptation --offline
python3 scripts/run-examples.py --id representation-classification --allow-network
python3 scripts/run-examples.py --id continued-pretraining --offline
python3 scripts/run-examples.py --id ner-complete --allow-network
```

Use `--allow-network` for a profile's first missing public-model files and `--offline` afterward. No account, API key, hosted inference, cloud tracing or remote model code is needed. `models.json` pins revisions, filenames, sizes and LFS SHA-256. Downloaded weights stay in ignored `.cache/e7/`; trained artifacts and executed notebooks stay in ignored `examples/artifacts/e7-*/`. Existing trusted public Hub cache files can be copied into the task cache and checked. Configuration/tokenizer files are pinned by immutable repository revision and checked for expected size; LFS weights additionally receive SHA-256 verification.

The isolated lock pins PyTorch 2.8.0, Transformers 4.57.1, Sentence Transformers 5.1.2, SetFit 1.1.3, NumPy 2.3.5, scikit-learn 1.7.2, datasets 4.3.0 and Accelerate 1.12.0. Linux resolves official CPU PyTorch wheels; macOS uses its platform wheel. These profiles use float32 CPU with one PyTorch thread. Tiny BERT is about 17.8 MB of weights; the pretrained sentence model additionally used for SetFit is about 70 MB. Tokenizers, configuration, package installation and multiple saved checkpoints require additional space.

Pinned sources are `prajjwal1/bert-tiny` at `6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837` (MIT model card), `google-bert/bert-base-uncased` tokenizer at `86b5e0934494bd15c9632b12f734a8a67f723594` (Apache-2.0), and `sentence-transformers/paraphrase-MiniLM-L3-v2` at `4ca70771034acceecb2e72475f72050fcdde4ddc` (Apache-2.0). No weights or book PDF are committed. Torch's supported checkpoint loader reads the pinned BERT weight artifact; remote Python code is disabled.

Actual duration, platform, package versions and child-process peak memory are in `../../verification/records/`. The runner's RUSAGE_CHILDREN peak is cumulative within a selected command and includes environment probes, so it is not an independent per-module memory profiler or a GPU comparison. All six author workflows complete in small CPU profiles; installation is separate. CUDA, MPS training, mixed precision, book-scale datasets and full benchmark configurations are unverified.

## Fresh actual notebooks

```sh
examples/training/representation/.venv/bin/python examples/training/representation/verify_notebooks.py
```

Use `--allow-network` for first downloads or repeated `--id` to select notebooks. Every notebook has provenance, prediction, actual shared-source training, error inspection, saved-result reload and transfer guidance. Each receives a fresh actual Jupyter kernel and executes every code cell in order. The environment includes pinned ipykernel, nbclient and nbformat; task-local Jupyter paths avoid modifying user kernel registrations.

`notebook-execution.json` records the full six-notebook pass, exact notebook/lock/verifier hashes and shared training-source hashes. Selected runs write a separate evidence file. Executed notebook copies remain ignored. This establishes actual kernel execution, not independent notebook-editor usability. Run notebooks **before** final CLI `--record`: both overwrite training artifacts and timing-bearing outputs.

The synchronized notebook generator is the authority for notebook source. Do not independently format generated notebook cells and retain their old execution hashes. After a notebook change regenerate its view, rerun its fresh kernel, then create final CLI records. Formatting shared Python sources likewise requires fresh verification.

## Data and independent evaluation units

The original MIT `embedding-training.json` has 78 service-request sentences: 24 train, 12 validation, 18 test and 24 separate unlabeled training-pool rows. Six service intents recur across splits, but exact canonical sentence text and IDs do not. Binary gold labels mean same requested service; they are not NLI entailment/neutral/contradiction or human-graded STS. Reserve/cancel phrasing supplies a useful lexical near miss. The 120 gold pairs include 72 train, 12 validation and 36 test pairs. The 120 candidate silver pairs are uniformly sampled without intent labels from the separate pool. Hidden author intents are used only afterward to diagnose teacher errors, never teacher training or candidate selection.

This is a small hand-authored same-domain teaching experiment, not a broad semantic benchmark. Labels, selected steps, metrics and data versions are explicit. Training text and pair IDs cannot cross split boundaries. Generalization to new intent classes, languages and natural collections requires new data.

L18/L19 reuse E4's UCI Sentiment Labelled Sentences IMDB subset: 200 train / 40 validation / 80 test balanced reviews. Source: Kotzias (2015), [UCI dataset DOI](https://doi.org/10.24432/C57604), CC BY 4.0. Existing manifests preserve archive/file hashes, exact-duplicate removal, deterministic stratified splitting and source changes. Challenge probes are excluded here. The reused test set was previously published in E4, but is excluded from new updates, masked-language adaptation and checkpoint/hyperparameter selection. No claim of a blind fresh external benchmark or contamination-free original pretraining is made.

Three fixed seeds (11, 23, 37) each select eight train reviews per class. Every L18 method gets identical selected IDs within a seed. Validation annotation cost is additional; sixteen training labels does not mean the entire experiment used only sixteen labeled records. Means, standard deviations and ranges describe three selections, not confidence intervals.

NER reuses the original E2 26/6/8 fixture, complete flat PER/ORG/LOC spans, disjoint name sets and shared templates. Empty-entity examples are included. The small fixture's high F1 does not establish natural-domain or CoNLL performance.

## L15: actual objectives, updates and reusable encoders

Three fresh tiny-BERT/mean-pooling encoders train with actual Sentence Transformers SoftmaxLoss, CosineSimilarityLoss and MultipleNegativesRankingLoss. This initializes from pretrained language weights, not random language-model training. Mean pooling respects the attention mask. Inputs use a declared 64-token cap; these original short fixtures fit it.

Softmax trains [u,v,|u−v|] → a two-class service-relation head; the optimizer includes that head and its changed weights are checked. Cosine loss is MSE against numerical labels, not universally 1−cos. The pinned MNR configuration is directional row cross entropy at scale 20, not automatically symmetric CLIP loss. Positive batches contain one pair per distinct intent, with duplicate/false-negative checks. Independent arithmetic matches actual library losses within declared float32 tolerances.

Transparent AdamW loops use learning rate 3e-4, weight decay .01, gradient clip 1, sixty steps, and eight paired rows or six distinct-intent MNR rows per batch. Validation is inspected at step 1 and every ten steps. The pair classifier minimizes validation pair cross entropy, cosine regression minimizes validation pair MSE, and MNR maximizes validation retrieval MRR; the first best checkpoint wins. Test data do not select steps. Actual gradients, changed-tensor groups, optimizer settings and saved artifacts remain recorded.

Pair MSE, Pearson/Spearman and retrieval MRR/Recall@1 evaluate the selected encoders. Constant correlation is explicitly undefined. One candidate per service and other unseen phrases define the original retrieval task; full 128-dimensional cosine determines rank. The visualization freezes a PCA fitted on baseline training vectors, then projects actual before/after test vectors. Two dimensions omit information. Stored transformer/tokenizer/pooling artifacts reload to equal embeddings. Objective state is separately saved when it contains training-only parameters.

Complete STSB/MTEB execution is not claimed. Describe benchmark/task versions, languages, data revisions and metric aggregation before comparing external reports; the book's historical MTEB counts are not timeless specifications.

## L16: full trained-teacher augmentation

A genuine tiny BERT cross-encoder trains on gold training pairs for sixty steps, with learning rate 3e-4 and validation binary cross-entropy checkpoint selection. It labels only separate training-pool candidates with continuous same-intent probabilities. Initial/selected hashes, first query gradient, actual teacher evaluation, artifact reload and source IDs are retained.

Gold-only and gold-plus-silver students share initialization, sixty-step cosine-regression budget and validation selection. Adding silver changes data distribution and gold exposure; it is not an isolated effect of label correctness. The original inspected run includes fourteen thresholded teacher disagreements and worse augmented-student pair MSE. The student receives continuous labels, not thresholded diagnostics. Imperfect calibration, class balance and teacher mistakes are valid explanations to investigate, not reasons to hide the result.

## L17: real TSDAE and subsequent adaptation

The actual DenoisingAutoEncoderLoss uses CLS pooling, seeded 60% whitespace-word deletion with at least one retained word, and one encoder vector of shape [batch,1,128] for decoder cross-attention. The shared compatible encoder/decoder parameters are genuinely tied; newly initialized decoder cross-attention is optimized. Its single-key softmax is always one, so query/key selection gradients are zero while value/output projections can learn. The script explicitly checks the actual value gradient, bridge changes, encoder gradients, tied embeddings and artifact reload instead of assuming every tensor must have a nonzero gradient.

Training is sixty steps, batches of four, AdamW learning rate 1e-4, weight decay .01 and clip 1. Fixed-corruption training reconstruction loss is a transparent probe, not a held-out language-model score. Actual validation retrieval MRR selects among trained checkpoints at steps 10–60; the unmodified baseline remains a separately reported control. The causal decoder sees gold previous tokens in training. Greedy reconstruction and teacher-forced next-token output are separately labeled. The pinned loss ignores padding target IDs; right padding lies after useful causal targets. CLS is a declared pooling choice, not a claim that contextual mean pooling has no positional information.

The run preserves repetitive free reconstruction and several intervals where reconstruction loss decreases without retrieval improvement. TSDAE-only, gold-only and TSDAE-plus-gold task results are shown. The latter two have the same supervised CLS-regression budget; the combined route has additional unsupervised work. Broad domain-transfer or out-of-domain capability preservation is not established.

## L18: freezing and genuine two-stage SetFit

For each seed, tiny BERT runs full adaptation, embeddings+layer-0 freezing, and head-only adaptation. Module names express the policy; no magic parameter index is used. Trainable counts, first gradients, changed tensors and exact frozen hashes verify it. The head-only representation is in eval mode while the head trains; frozen parameters alone do not imply dropout is disabled. Sixty-step BERT runs select the first maximum validation macro-F1 at ten-step checkpoints.

The same sixteen reviews feed TF-IDF/logistic, frozen MiniLM sentence embeddings/logistic and real SetFit. SetFit uses the actual pinned SetFitModel and ContrastiveDataset sampler, actual Sentence Transformers cosine regression, a transparent CPU AdamW loop (twenty-four steps, eight pairs, 2e-4, weight decay .01, clip 1), then SetFitModel.fit for the logistic head on the original labeled reviews. Generated/distinct pair rows are recorded; they do not represent new independent human labels. Saved SetFit body/head artifacts reload on CPU and reproduce predictions.

SetFit 1.1.3's convenience Trainer implicitly selected MPS on the author Mac even when the initial model was on CPU. That path failed the explicit CPU assertion and is not the verified profile. Using the real library sampler/model with a visible CPU optimization loop preserves the two-stage algorithm without monkey-patching accelerator internals. Verified results include actual device, first attention gradient and changed body hash. Do not relabel the frozen-vector control as SetFit. BERT versus MiniLM comparisons also change architecture and prior training; the frozen-sentence control is the direct comparison for the embedding-adaptation stage.

## L19: continued MLM with downstream controls

The actual Transformers collator selects eligible tokens at probability .15, then nominally replaces 80% with [MASK], 10% with a random vocabulary ID and 10% unchanged. All selected positions target the original ID. Original special tokens, padding and nonselected targets receive −100. Observed unchanged tokens cannot identify the exact random branch. Each actual batch must include a target; otherwise a mean masked loss is undefined.

Only the 200 train review texts enter updates, without their task labels. Validation uses fixed masks (seed 999) and token-count-weighted loss; training uses separate dynamic masks (seed 42). Two-step smoke and sixty-step evaluated profiles start fresh. AdamW uses 1e-4, .01 weight decay, clip 1 and batches of eight; the first minimum validation masked-loss checkpoint is saved/reloaded. Fill-mask top candidates are actual output, not downstream evidence.

Downstream classifiers use identical data, initialization seed, sixty-step optimizer budgets and validation selection. The MLM checkpoint omits the original BERT pooler: both classifiers therefore start from the same base pooler/head and only shared adapted embeddings/encoder weights are transferred. Their initial pooler/head hashes must agree. Training RNG is reset after construction so loading an additional encoder does not alter dropout draws. These controls avoid attributing a random new head/pooler to domain adaptation. Test reviews never enter pretraining. The result remains a small fixed experiment, not universal evidence that continued pretraining helps.

## L20: extend and reuse the E2 NER implementation

The E2 main function now accepts optional model/tokenizer/artifact paths and returns its result; default behavior remains the reference workflow. E7 invokes that same ninety-step trainer with pinned local snapshots, preserving frozen embeddings, validation selection, full-sentence decoding and strict entity metrics. Its E2 execution record is refreshed after the source change. No second training loop is copied.

Actual tokenizer probes show split names, multiword entities, no-entity input, specials, padding and truncation. First-only supervision is the trained route. All-subtoken B→I propagation is a separately checked alignment alternative, not another trained model. Both reject truncated training records; an inspection mask can display retained pieces without approving shortened entities. Strict typed spans retain sentence identity and exclusive end offsets. Invalid predicted I tags use the explicitly shared repair policy and are separately flagged. Constructed correct/boundary/type/invalid-I cases are distinct from actual held-out model outputs.

Do not pass individual tokens as separate metric sentences. It destroys multiword spans and can inflate a misleading score. Flat BIO excludes nested entities and needs domain-specific annotation guidelines before use on another collection.

## Verification and extension

Run `python3 -m unittest discover -s tests/examples -v`, `npm run test:training`, and `npm run release:build`. Contracts cover stable cross entropy, directional MNR, split/duplicate/false-negative guards, seeded deletion, subtoken policies, truncation, actual teacher provenance, exact freezing, actual CPU SetFit updates, equal downstream pooler/head initialization, MLM label masks, single-key TSDAE gradients, saved predictions and fresh-kernel source hashes. Node/Python arithmetic agree; browser float64 versus tensor float32 reductions use explicit tolerances.

After changing a source, shared helper, lock or fixture, regenerate notebooks, run actual fresh kernels, then final CLI `--record` commands. Synchronize published snapshots, rerun affected offline evidence calculations with new records, reassess source/audit hashes, and run the release gate. Never edit a failed record or attach an old badge to a changed artifact. No unsupported quality threshold turns a bad model result into a passing claim; model assertions check execution contracts and intended updates.

Original explanations and code follow the pinned book notebooks at ea3390819997999a51983677b80b3aac4dc50ada. [Sentence Transformers training/loss documentation](https://www.sbert.net/docs/sentence_transformer/training_overview.html), [SetFit source](https://github.com/huggingface/setfit/tree/v1.1.3), and [Transformers token classification](https://huggingface.co/docs/transformers/v4.57.1/en/tasks/token_classification) provide the primary API context. Modernization corrects pretrained-versus-random initialization, nonindependent pair counts, MLM epoch/selection ambiguity, meaningful module-name freezing and sentence-preserving NER metrics. Independent technical/editorial review, beginner acceptance, fresh external benchmarks and GPU verification remain pending.
