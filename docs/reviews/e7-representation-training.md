# E7 representation training: author technical pass

Date: 2026-11-24. Reviewer: implementing assistant (author technical pass). Independent technical/editorial review and beginner pilot remain pending. Execution, coverage and teaching acceptance are separate; no independent acceptance or benchmark results are invented.

## Source and scope

Fifteen Chapter 10–11 lessons are expanded, including new Augmented SBERT. Each has specific definition, theory, worked example, implementation, use cases, limitations and topic-specific self-checks. Existing URLs/completion IDs are retained. Losses and correlation/benchmark proposals resolve to canonical anchored sections. E7 adds L15–L20, a service-search adaptation project, a dedicated encoder-training route, and an updated applications route.

Chapter 10–11 extracted text and pinned official notebook workflow stages were inspected. Representative PDF pages 315, 321, 327, 335, 347, 353, 365 and 371 were visually inspected. The official notebook bytes matched the existing pinned hashes. Original prose, data and visuals replace book text/artwork. No PDF or model weights are redistributed.

The adaptation makes pretrained initialization explicit, distinguishes binary service relation from NLI/STS, uses module names rather than fragile freeze indices, separates pair count from independent labels, preserves whole-sentence entity metrics, and distinguishes numerical target contracts. Contextual mean pooling is not described as losing all position information. Complete MTEB/STSB, MNLI/CoNLL and book-scale GPU runs remain unverified.

## Authentic model and data evidence

All six entrypoints perform actual training on pinned public models, using an isolated float32 one-thread CPU environment. Actual gradients, changed weights, exact frozen hashes where applicable, validation selection and saved-artifact reload supplement task metrics. Six staged notebooks execute every code cell in fresh actual kernels; notebook, lock, verifier and shared-source hashes bind their evidence. Final CLI records follow notebook artifact writes. Published snapshots must match recorded artifact hashes.

The original intent fixture has 24/12/18 train/validation/test sentences and 24 separate unlabeled training-pool rows. Same intent is an explicitly authored service relation. Exact text/IDs do not cross splits. A trained cross-encoder teacher produces continuous silver labels only for the training pool. The 14/120 thresholded teacher disagreements and worse augmented-student similarity error remain visible. Equal student steps do not imply equal gold exposure. The observed data are a small mechanism experiment, not a broad semantic benchmark.

Actual Sentence Transformers pair softmax, cosine regression and directional MNR losses are independently checked against arithmetic and task-matched validation rules. Pair-head updates are checked as well as encoder updates. Full-dimensional rankings determine quality; a PCA fitted only on baseline training vectors is a limited explanatory projection.

Actual TSDAE uses CLS pooling, seeded deletion, tied compatible parameters and a one-vector decoder condition. A one-key attention softmax has zero query/key selection gradients; value/output paths still learn and are explicitly checked. Free repetitive reconstruction, teacher forcing, falling reconstruction loss without retrieval gains and adaptation-only/gold-only/combined task results are retained. Surrogate improvement is not task acceptance.

L18 repeats eight-label-per-class selection under three seeds and shares IDs across six methods per seed. It exposes full/partial/head-only BERT, lexical and frozen-vector baselines, and real two-stage SetFit with its actual library sampler/model/loss and a transparent CPU optimizer. The convenience Trainer's implicit MPS path failed the CPU assertion and is not claimed verified. The final saved model/body/head reload explicitly uses CPU. Architecture/prior-training differences and validation annotation cost are disclosed.

L19 performs genuine MLM only on the 200 train review texts, with dynamic training and fixed validation masks. A two-step smoke run is separate from the evaluated sixty-step profile. Selected unchanged tokens retain targets; original specials/padding are ignored. Equal downstream head/pooler hashes and reset training randomness control an otherwise easy-to-miss initialization confound when loading an MLM checkpoint. The previously published E4 test reviews remain excluded from fitting/adaptation/selection; they are not a fresh blind external benchmark.

L20 reuses the E2 trainer through optional explicit paths and a returned result, with its original default behavior and a refreshed E2 source execution record. Actual alignment probes cover split words, multiword entities, empty-entity input, specials, padding and truncation. The trained route uses first complete subtoken supervision; B→I propagation is a separately verified alignment alternative. Whole-sentence strict spans, invalid-BIO policy, real held-out predictions and constructed error cases remain distinct.

## Review packet and remaining work

Reviewers should explain one wrong-service rank, one teacher/silver error, one loss/task disagreement, a freezing-mode distinction, a selected-but-unchanged MLM token, and a multiword NER boundary error using source IDs and actual intermediate evidence. Check that the UI distinguishes calculations from saved inference/training, that loss axes/denominators and selected steps are understandable, and that artifacts include the tokenizer/pooling state used for evaluation.

Use `examples/training/representation/README.md` for setup, exact configurations, provenance and reproduction. The release gate covers source/lock/support/fixture/artifact drift, offline calculations, behavioral contracts, Astro compilation and built links. Independent teaching acceptance, a beginner pilot, broad external benchmarks, out-of-domain preservation and GPU configurations require separate evidence. The coverage inventory retains its acceptance gates rather than labeling author work independently accepted.

## Final release verification

The complete release gate passed: 98 behavioral tests (57 Python and 41 Node), 201 offline examples, 234 registry entries, zero Astro diagnostics, 150 built pages and 13,905 checked internal links/assets. All six actual CPU training CLI records and all six fresh-kernel notebook records passed, with source/lock/artifact hashes. Desktop 1280×900/light and mobile 390×844/dark QA exercised keyboard loss controls/reset, objective/seed/classifier selectors, teacher-disagreement provenance, training and validation curves, fixed-PCA evidence, empty MLM target state, disabled special tokens, propagated/first-only alignment, safe truncation rejection and no-entity input. Graph alias “gold and silver pairs” resolves to Augmented SBERT with SBERT parent and fine-tuning prerequisites; site search returns the new lesson.

Browser testing caught mixed dynamic text children in SVG title elements causing hydration mismatch. Titles now render as single text strings; the production build was rebuilt and all five chart-based explorers hydrated without fresh console errors. Browser QA is an author pass, not independent teaching acceptance. A source graph prerequisite was corrected to the existing canonical encoder-only concept rather than inventing a BERT page ID.
