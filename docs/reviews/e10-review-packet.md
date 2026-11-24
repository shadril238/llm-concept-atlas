# E10 review packet

Prepared 2026-11-25 (Asia/Seoul). Status: evidence reconciliation and review preparation; independent acceptance pending. This document records author preparation, not an independent review. A reported pass without its scope, reviewed revision and evidence is not converted into acceptance.

## What reviewers receive

- [Source-objective inventory](../../src/data/book-sections.json): 231 source entries, including 160 counted leaf objectives, with book locations and destination anchors.
- [Generated gap report](../audits/book-gap-report.md) and [lesson audit](../../src/data/lesson-audit.json): remaining work and per-lesson source hashes.
- [Machine evidence inventory](./e10-evidence-audit.json): chapter counts, missing practical records, indirect evidence references and input hashes. Regenerate with `npm run evidence:report`; the release build rejects stale output.
- [Example registry](../../examples/manifest.json) and [verification report](../../examples/verification/README.md): executable entry points, locks, models, fixtures, scope and recorded results. Existing checks validate source/contract/lock/artifact hashes; the new inventory additionally checks that direct objective references point to registered passing records.
- [Author reviews E2–E9](./), [teaching rubric](../../enhancement.md#33-editorial-rubric), [lesson template](../templates/reference-lesson.md), [candidate notes](../releases/release-candidate.md), and [blank review record](../templates/e10-review-record.md).

All 160 leaf objectives have destinations; none yet has recorded independent acceptance. Of 51 practical objectives, 49 have recorded execution in the declared atlas configuration. These counts do not certify GPU, book-scale performance or teaching quality. Twelve practical objectives currently reference the E5 author summary indirectly; the machine inventory lists them. Reviewers must follow the summary to the specific workflow records and check the claim rather than treat the summary as a raw execution record.

## Reconciliation completed in this change

Eight E3-related practical objectives retained `not-verified` despite existing passing runs. Their source, pinned assets, artifacts and objective wording were compared; only execution metadata was reconciled. Existing record bytes, dates, model code and independent review statuses were preserved. This is not eight new model runs.

| Objective | Existing run | Scope checked |
| --- | --- | --- |
| `book-ch01-first-generation` | `first-generation` | Matching tokenizer/model; continuation IDs, shapes and generation settings. |
| `book-ch02-run-llm` | `first-generation` | Pinned CPU Pythia-14m tokenization and generation. |
| `book-ch02-tokenizer-comparison` | `tokenizer-comparison` | Two tokenizers, four identical inputs including Korean/emoji and code with whitespace and the numeral 2. This is a small probe set. |
| `book-ch02-pretrained-word-vectors` | `representation-inspection` | Checksum-verified pretrained GloVe; original-space neighbors and the lesson's warning that contextual similarity does not imply synonymy. |
| `book-ch02-song-recommendations` | `playlist-embeddings` | Original playlist/catalog fixture, learned retrieval, train-only popularity baseline and held-out metrics. |
| `book-ch02-song-training` | `playlist-embeddings` | Actual skip-gram updates, train/validation/test separation, cold-item boundary and saved-vector reload. |
| `book-ch03-inputs-outputs` | `first-generation` | Actual input IDs, hidden states, LM-head/logit agreement and new tokens. |
| `book-ch03-kv-cache` | `cache-equivalence` | Float64 CPU logit equivalence before same-token timing and cache tensor-byte measurement. |

Each objective now has an `executionScope` explaining limits. The tokenizer record is reused on the basis of actual probes, not its broad source tags. Pretrained-word-vector evidence comes from loaded GloVe, not the playlist model's broad tags. Small synthetic recommendation data does not establish real-user usefulness; cache tensor bytes do not represent total process memory. Two exact duplicate API-RAG destinations were removed without removing any unique destination. E9's dated 41/51 snapshot remains historical; the current dashboard uses the reconciled inventory.

## Independent technical and editorial review

Work in chapter-sized batches. For each counted objective, open its exact destination, source location and declared evidence. Review parent/context entries too, recording why a contextual decision activity is appropriate when code is not required. Do not infer acceptance for an entire chapter from a few sampled lessons.

1. Record reviewer identity/role, date, independence from the author, reviewed Git commit and objective IDs. If feedback was supplied elsewhere, retain its link and exact scope rather than inventing scores from a general approval.
2. Follow the explanation as a learner: definition, theory, prerequisites, worked intermediate values, visual controls, implementation, evaluation, applications, limitations and reasoned exercises. Check every required coverage dimension separately.
3. Re-derive equations, dimensions and reductions. Check source cautions already recorded in the inventory: correlation can be negative; weighted F1 differs from macro-F1; LoRA parameter count depends on consistently named shapes; GGUF is a container rather than a precision.
4. Inspect train/test separation, token/label alignment, attention and loss masks, frozen/trainable tensors, baseline fairness, invalid-output accounting and save/reload or merge tolerances. Separate optimization correctness from task quality. A correctly executed weak model can still fail its intended application.
5. Run affected examples in their documented locked environment. First-download network permission and hardware requirements come from that workflow's guide. Retain command, commit, platform, pins, assertions and raw result records. Changes to code, locks, fixtures, support files or contracts require new affected execution evidence; documentation-only reconciliation does not justify manufacturing a new run.
6. Score all ten rubric criteria, 0–3 each. Proposed pass: at least 24/30, no score below 2, no unresolved factual or executable error; cite evidence for every score below 3. Record required revisions and retest their outcomes.
7. Add the signed-off review to `src/data/review-records.json`, backed by the actual review document. Update the relevant objective's `acceptanceRecord` with reviewer/date/evidence, its dimension findings and mappings, and only then its review status. Accepted practical objectives still require supported execution evidence. Reassess affected lesson audits/hashes after actual content changes, then regenerate both reports.

The blank record is a template, not proof that any review happened. Do not set `independentReview: true`, `accepted: true` or dimension sufficiency while the associated evidence is missing.

## Learner trials

Run separate beginner and intermediate sessions with readers who did not author the lessons. Record starting knowledge, selected objective IDs, reviewed revision, browser/device, tasks, interventions, observed misconceptions and follow-up fixes. Use a pseudonym where preferred; no personal contact information is needed.

| Route | Task and observable outcome |
| --- | --- |
| Beginner: tokens to model output | Find tokenization using search; predict how whitespace affects a tokenizer; inspect actual pieces/IDs; explain lookup rows versus contextual states; identify prompt versus continuation. |
| Beginner: graph and prerequisites | Find KV cache, follow an explained prerequisite, distinguish parent/prerequisite/related edges, and explain what cache memory includes. |
| Intermediate: semantic search to RAG | Retrieve a grounded answer and an absent-evidence case; trace citation IDs; explain retrieval versus generation evaluation and why a citation alone does not prove support. |
| Intermediate: adaptation | Follow the instruction-to-preference route; inspect masked target tokens and frozen weights; distinguish adapter optimization, held-out quality, merge equivalence and CPU NF4 versus GPU QLoRA. |
| Both: discovery and accessibility | Complete search with keyboard only, open/close the dialog and regain focus, follow a practice destination, retain reading/practice after reload, and inspect a narrow screen, zoom, dark theme and text alternatives. |

Ask learners to predict before changing a control, explain the result in their own words and solve one new example without the solution open. Record actual completion and help required; no invented pass rate or imposed time target. A failed task becomes a specific lesson/UI revision with a follow-up attempt. Sampled pilots inform revisions but do not replace review of all required objectives. Include assistive-technology checks with the actual browser/tool/version used; author keyboard checks are not formal screen-reader acceptance.

## Open execution and release decisions

- `book-ch04-chatgpt`: optional hosted classification remains a documented comparison protocol plus offline importer, as requested. No provider execution is claimed.
- `book-ch08-api-rag`: optional hosted grounding remains documented/imported only; live provider source/citation objects remain unverified.
- E8 GPU QLoRA/paged optimization: only Linux CPU NF4 evidence exists. This requires a supported GPU implementation/profile and genuine run evidence; changing a CPU record's device field is invalid. Follow the [generation guide](../../examples/training/generation/README.md) for current CPU scope. A future GPU profile needs its own dependencies, hardware limits, masks, trainable-weight checks, memory measurements, held-out evaluation, artifacts and reload/merge checks.
- Three legacy examples remain unverified in the registry. Their successors have separate records; do not transfer badges to historical scripts or imply the official book notebooks were executed. The official-notebook manifest remains `not-run`.
- Independent technical/editorial acceptance and learner records have not been supplied in the repository. External resource reachability and model/dataset-license terms need a dated final review; internal link checks do not establish that every third-party URL is reachable.

These limitations stay visible. The two hosted workflows remain part of the book inventory; their optional implementation policy does not silently waive the original E10 gate. A release can be labeled a limited preview with these limits, but it cannot be called full enhancement acceptance.

## Release procedure after evidence arrives

1. Attach review and learner records to the reviewed revision; address findings, rerun affected examples and update objective-level acceptance. Preserve the original run dates and historical author reviews.
2. Check original-asset attribution, data/model manifests and licenses, primary-source links and actual notebook/download entry points. Record dated external-link results and redirects/failures; retain source pins rather than switching versions merely to get a response.
3. Run `npm run coverage:report`, `npm run evidence:report` and `ASTRO_TELEMETRY_DISABLED=1 npm run release:build`. Keep the reviewed commit and CI run URLs in the release notes. Generated evidence reports must be committed with their source changes.
4. Check the deployed search, graph, selected concept, chapter coverage, notebook downloads and reading/practice flows. Log the exact deployment revision and browser/device. Author E9 browser evidence is linked in its review, not relabeled as a fresh E10 independent session.
5. Resolve every original E10 gate or explicitly publish a limited preview. Fill the candidate notes with actual evidence, then create a reviewed release/tag only when its stated gates are met. No reviewed release/tag has been created by this preparation change.
