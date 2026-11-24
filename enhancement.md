# LLM Concept Atlas: teaching depth and book coverage enhancement plan

**Prepared:** 2026-11-18  
**Last updated:** 2026-11-19  
**Status:** E0 source inventory and initial lesson audit, and E1 infrastructure implemented; E2–E7 implementation delivered, independent review pending; E8 CPU implementation delivered with GPU/review gates open; E9–E10 pending. See the [gap report](./docs/audits/book-gap-report.md) and [audit guide](./docs/audits/README.md).
**Baseline:** Repository commit `a6b3ec4`, following the original Phases 1–5 in [plan.md](./plan.md).  
**Source edition:** _Hands-On Large Language Models_, Jay Alammar and Maarten Grootendorst, first edition, September 2024.  
**Primary goal:** A learner can explain, trace, implement, evaluate, and apply the book's ideas without depending on unexplained terminology or disconnected code snippets.

This document is the implementation contract for the next stage. It covers missing subjects **and the teaching quality of existing subjects**. A page labeled `full`, a passing build, and a working toy calculation are useful starting points; none alone proves that a concept has been taught thoroughly.

The chapter backlogs below come from the supplied PDF's detailed contents and targeted checks of its discussion, alongside inspection of the current repository. They are an actionable starting inventory. Enhancement E0 now records 231 source entries, audits all 96 existing concept lessons, and links a pinned snapshot of all 12 official notebooks. It is an initial coverage/depth audit, not independent teaching acceptance or model execution verification; exhaustive completion remains pending.

## How to use this plan

1. Complete **E0–E2** first: establish an honest inventory, build the example infrastructure, and finish three reference lessons.
2. Implement chapter batches **E3–E8** in dependency order. Every batch includes explanations, code, visuals, exercises, graph connections, and review.
3. Complete **E9–E10** for the integrated learning experience and release review.
4. Use the chapter specifications as work queues, the example contracts as acceptance criteria, and the lesson rubric as the editorial gate.
5. Update the checklist at the end with evidence as work finishes. All new paths, fields, components, scripts, and commands described here are **proposed**, unless explicitly identified as existing.

Do not interpret this as a request to replace the application stack. Keep Astro, Starlight, MDX, React islands, Cytoscape, Pagefind, and GitHub Pages. Concentrate effort on learning value and reproducible implementation.

### Navigation

- [Completion criteria](#1-what-completion-should-mean) and [current gaps](#2-current-baseline-and-problems-to-fix)
- [Teaching standard](#3-teaching-standard-for-every-lesson) and [coverage system](#4-build-a-trustworthy-coverage-system)
- [Code example standard](#5-code-examples-implementation-standard)
- [Interactive UI, graph, and search](#6-interactive-lesson-graph-and-search-experience)
- [Chapter-by-chapter backlog](#7-chapter-by-chapter-implementation-backlog)
- [Detailed LoRA, NER, and BERTopic specifications](#8-three-precise-examples-of-the-required-depth)
- [Projects and learning paths](#9-applied-projects-and-learning-paths)
- [Implementation phases E0–E10](#10-implementation-phases-dependencies-and-completion-gates)
- [Per-change workflow](#11-per-change-workflow-and-review-packet), [risks](#12-risks-and-concrete-responses), and [acceptance checklist](#13-final-acceptance-checklist)
- [Sources](#14-sources-and-implementation-references) and [implementation tracker](#15-implementation-tracker-and-immediate-next-task)

## 1. What completion should mean

A learner who completes a lesson should be able to:

- **Define:** Give a precise explanation in their own words and distinguish the concept from a nearby alternative.
- **Trace:** Follow an input through intermediate representations, calculations, and outputs.
- **Predict:** Explain what should change when one important input or parameter changes.
- **Implement:** Run a suitable example from a clean environment and modify it for a related problem.
- **Evaluate:** Choose a relevant metric, inspect a failure, and recognize when the method is unsuitable.

The enhanced atlas is complete only when:

- Every substantive section of the selected book edition has an explicit, reviewed mapping to a lesson, anchored subsection, lab, or contextual note.
- Every mapped learning objective has sufficient explanation and evidence. A link to the authors' notebook counts as a source, not as an implemented atlas lab.
- Every major practical workflow has an atlas-owned executable implementation, a baseline, evaluation, and recorded execution evidence on its supported environment.
- Existing lessons have been reviewed against the new rubric, including those already labeled `full`.
- A learner can follow prerequisites, search synonyms, and navigate relationships without dead ends or duplicate definitions that contradict each other.
- Missing, partial, unsupported, historical, and supplemental material is labeled honestly.

### 1.1 Scope boundaries

| Scope                        | Treatment                                                                                                                                                                        |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Book core                    | All substantive chapter sections and important embedded concepts; required for book coverage.                                                                                    |
| Book context                 | History, model access, practical setup, responsible use, and architectural research directions; explain them appropriately without forcing a training notebook onto every topic. |
| Prerequisite bridge          | Vectors, matrix shapes, softmax, gradients, train/validation/test splits, metrics, and Python basics needed to follow the book; add short original bridges where needed.         |
| Implementation modernization | Recreate the book's learning objective with supported libraries; record differences from the 2024 implementation.                                                                |
| Supplemental extension       | Useful additions such as a hybrid retrieval comparison or broader deployment advice; clearly mark them as extensions rather than claiming the book teaches them.                 |
| Outside this project stage   | Training a foundation model at scale, a paid hosted inference service, user accounts, a general chatbot tutor, and unrelated agent frameworks.                                   |

Preface/setup guidance and the afterword receive contextual coverage where useful. Bibliographic entries, index terms, repeated summaries, and figure captions are inspected for concepts; they do not automatically become separate lessons or inflate the coverage denominator.

## 2. Current baseline and problems to fix

The inspected repository contains 96 concept lessons, a curated map of 95 chapter-topic entries across 12 chapters, 11 browser teaching labs, three paths, and two use cases. These are repository counts, **not a count of every concept in the book**.

| Existing asset                                       | Keep                                                        | Enhancement needed                                                                                                                                           |
| ---------------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `src/content/docs/concepts/`                         | Stable lesson URLs and useful introductory explanations     | Audit every lesson for depth, correctness, distinctive examples, missing prerequisites, and actual implementation.                                           |
| `src/data/book-map.ts` and `topic-coverage.json`     | Chapter navigation and existing topic mappings              | Add a source-section inventory; the existing curated map cannot establish exhaustive book coverage.                                                          |
| `ConceptPrimer.astro` and `ConceptConnections.astro` | Lesson introduction, progress, connection navigation        | Add objectives, honest review/implementation status, prerequisite reasons, and links to relevant labs.                                                       |
| `LessonFlow.astro`                                   | Simple process overviews where appropriate                  | Add concept-specific visuals showing actual quantities and transformations. Three generic boxes are insufficient for attention, loss functions, or training. |
| Existing simulation components                       | Account-free interaction and clear illustrative-data labels | Separate simulation logic from presentation; add meaningful experiments, intermediate values, and text equivalents.                                          |
| `tests/content.test.mjs`                             | Structural and internal-link checks                         | Replace exact 96/95 counts and word/heading-count claims of completeness with inventory checks and review evidence.                                          |
| `scripts/check-code-examples.mjs`                    | Execution of 76 deterministic snippets                      | Replace regex-only discovery and exact-count assumptions with a manifest covering all examples, including the earlier lessons.                               |
| `src/lib/graph.js`                                   | ID, destination, parent, and prerequisite validation        | Add relationship explanations, aliases, and richer topic navigation while preserving stable IDs.                                                             |
| GitHub Pages workflow                                | Existing release build and deployment                       | Add appropriately separated example verification; keep heavy model training out of ordinary site deployment.                                                 |

Concrete examples of the depth problem:

- The current NER snippet pairs supplied tokens with supplied BIO labels. Keep that as a notation introduction, then teach subword alignment, prediction, training, and entity-level evaluation.
- The current BERTopic snippet counts words in preassigned groups. It does not demonstrate embedding, clustering, c-TF-IDF, or fitting BERTopic. Replace the main implementation with a genuine modular pipeline.
- The current LoRA example adds a small matrix update. Extend it to explain dimensions, scaling, initialization, gradient updates, parameter counts, save/reload, and adapter merging.
- Repeated prompts such as “identify what is prepared, what changes” should become topic-specific questions with meaningful expected answers.

Correct the broad coverage wording in `README.md`, `plan.md`, and `about-and-sources.mdx` during E0. Preserve the historical phase record, but describe the existing release as a curated chapter companion awaiting the deeper audit.

## 3. Teaching standard for every lesson

### 3.1 A lesson's teaching sequence

Use the sequence below as an editorial guide. Related sections may be combined when that improves reading; do not manufacture filler to meet a section count.

| Section                  | Required content                                                     | Evidence of quality                                                                              |
| ------------------------ | -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| The problem              | A concrete situation that motivates this concept                     | The learner understands why the concept exists before seeing terminology.                        |
| Learning objectives      | Three to five observable abilities                                   | Verbs such as calculate, compare, implement, diagnose; avoid “understand” as the only objective. |
| Prerequisites            | Required ideas, why each matters, quick recall prompt                | A reader can identify and repair a missing foundation.                                           |
| Definition               | Precise definition plus an accessible explanation                    | Explain the boundary: what counts and what does not.                                             |
| Intuition                | One useful mental model or analogy                                   | State exactly where the analogy stops being accurate.                                            |
| Mechanism                | Ordered steps and intermediate representations                       | Explain shapes, units, assumptions, and changing versus fixed quantities.                        |
| Mathematics              | Relevant equations with all symbols defined                          | Work a small numerical example; connect the equation to code.                                    |
| Worked example           | One complete trace from input to output                              | Show actual intermediate values, not a restatement of the pipeline.                              |
| Visual experiment        | A labeled visual and a meaningful intervention                       | Ask for a prediction before revealing the result.                                                |
| Implementation           | A runnable calculation and/or real workflow appropriate to the topic | Setup, inputs, outputs, tests, interpretation, and failure cases are available.                  |
| Decisions and use cases  | Specific task, data needs, baseline, tradeoffs                       | Explain when to choose the method and when to choose another.                                    |
| Failure analysis         | At least one concrete failure and misconception                      | Show how to detect the failure, not just warn that errors exist.                                 |
| Practice                 | Recall, application, debugging, and transfer where appropriate       | Include hints, worked solutions, and reasons wrong answers fail.                                 |
| Connections and evidence | Related concepts with reasons, source location, primary references   | The reader knows what to learn next and what each claim is based on.                             |

Foundational and complex lessons may need substantial length. Do not impose a word quota as a substitute for teaching. A contextual topic such as open versus hosted models needs a decision exercise; a fabricated Python snippet would add little value.

### 3.2 Writing and mathematical rules

- Introduce terms before using them; expand acronyms on first use and provide a searchable alias.
- Keep a consistent example through intuition, equations, visual, and implementation. Show explicit transitions when switching from invented data to real model output.
- Define each mathematical symbol, its dimensions, and its role. For a matrix multiplication, annotate both input shapes and the output shape.
- Separate model parameters, hyperparameters, input data, predictions, and evaluation labels.
- Explain logits versus probabilities, normalized similarity versus confidence, and token-level versus sequence-level quantities wherever relevant.
- Use original explanations and diagrams. Cite the book's chapter and printed page or section, then link primary papers and official documentation for technical details.
- Do not imply a generated explanation reveals a model's hidden internal reasoning. Teach observable outputs, explicit algorithms, and verifiable task performance.
- Do not present attention maps as a complete causal explanation, a 2D projection as the original embedding geometry, or a high metric as general reliability.
- Explain model-specific behavior as model-specific. Avoid claims that every tokenizer, transformer, or training method behaves identically.

### 3.3 Editorial rubric

Score each criterion from **0 to 3**: 0 = missing/incorrect; 1 = named but insufficient; 2 = independently usable explanation; 3 = explanation plus a convincing worked example, counterexample, or transfer task.

1. Definition and motivation.
2. Prerequisite accessibility.
3. Mechanism and mathematical accuracy.
4. Worked-example quality.
5. Implementation usefulness and reproducibility, or an appropriate decision activity for contextual topics.
6. Visual clarity and honest interpretation.
7. Use cases, alternatives, and limitations.
8. Exercises, feedback, and transfer.
9. Source traceability and distinction between book content and extensions.
10. Accessibility and navigation.

**Proposed pass gate:** at least 24/30, no criterion below 2, and no unresolved factual or executable-code error. Scores are review aids, not proof of learner success. Reviewers must cite a specific example for any score below 3. A first author pass is not an independent technical review.

### 3.4 Three reference lessons before scaling

Finish these end to end before upgrading the rest:

1. **Tokenization:** real tokenizer comparison, special tokens, multilingual cases, byte/subword distinctions, executable outputs, and a prediction exercise.
2. **LoRA:** a hand-worked matrix update, tiny trainable example, actual adapter workflow, scaling and dimensions, and merge verification.
3. **Named-entity recognition:** annotation policy, token/subword alignment, supervised training, span decoding, and entity-level metrics.

These deliberately cover a foundation, a mathematical mechanism, and a data-dependent training workflow. Keep the layout reusable but the explanations and questions specific to each subject.

## 4. Build a trustworthy coverage system

### 4.1 Source inventory

Create `src/data/book-sections.json` as the canonical source inventory. Give each entry a stable ID, such as `book-ch10-augmented-sbert`, independent of its display title.

For each entry record:

- Chapter, source heading, printed page range, and verified PDF page mapping.
- `scope`: `core`, `context`, `prerequisite`, or `extension`.
- `kind`: concept, workflow, contextual discussion, or structural container.
- Parent source section and a short original description of the learning objective.
- Required coverage dimensions: explanation, worked example, visual, real implementation, evaluation, and exercises. Record a rationale for dimensions marked not applicable.
- Destination lesson IDs and section anchors; lab/example IDs when applicable.
- Mapping status: `unmapped`, `partial`, or `mapped`.
- Separate review status: `unreviewed`, `changes-required`, or `accepted`.
- Remaining gaps, review record, and evidence references.

The PDF is a local audit source and must not be committed. Published source metadata should identify the book edition and printed locations, not the user's private filesystem path.

### 4.2 Audit procedure

1. Transcribe the detailed contents into an inventory; use source headings for identification and original text for explanations.
2. Read each section, inspect relevant figures, and inspect the corresponding official notebook. Add significant concepts and procedures that the contents omit.
3. Mark overview containers separately from leaf learning objectives. Repeated mentions point to the same canonical concept.
4. Map each objective to an existing page and exact anchor, a proposed new page, or a lab.
5. Assess depth against Section 3; mark partial coverage even if a page exists.
6. Verify the inventory against all 12 chapters. Keep an explicit list of unresolved items.
7. Generate a report by chapter and coverage dimension. Review mappings before upgrading a public coverage claim.

Do not automatically mark a whole section complete because a lesson contains its name. Do not count both a parent heading and all its children as independently completed concepts.

### 4.3 Separate three kinds of status

| Status family          | Meaning                                                                  |
| ---------------------- | ------------------------------------------------------------------------ |
| Source coverage        | Does an atlas destination address the book's actual learning objective?  |
| Editorial review       | Has someone checked accuracy, clarity, depth, and exercises?             |
| Execution verification | Has the applicable example run successfully in its declared environment? |

Expose useful public labels such as **Explanation available**, **Interactive demonstration**, **Runnable model lab**, and **Verified on [date/environment]**. These must derive from evidence, not author optimism. A historical example or an unverified GPU run remains visible with its limitation stated.

### 4.4 Schema changes and migration

Extend `src/content.config.ts` gradually with validated fields such as `learningObjectives`, `aliases`, `bookSections`, `exampleIds`, `reviewStatus`, and `reviewRecord`. Add `kind: lab` and a contextual-note type if needed.

- Preserve current `conceptId` values and URLs.
- Keep `lessonStatus` temporarily for compatibility; do not silently equate legacy `full` with a new review pass.
- Mark existing lessons `unreviewed` under the new standard until audited.
- Keep one canonical definition per concept. Reuse anchored explanations rather than creating competing pages for every spelling.
- Generate chapter coverage views from the canonical inventory. Retire the old independent coverage map only after its consumers have migrated and parity checks pass.
- Validate unknown IDs, missing anchors, duplicate aliases, nonexistent example IDs, invalid review records, and orphaned source sections.

## 5. Code examples: implementation standard

### 5.1 Three complementary levels

| Level               | Purpose                                                   | Requirements                                                                                                                       |
| ------------------- | --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| A: Mechanism        | Make the calculation visible                              | Small inputs, transparent code, intermediate values, appropriate assertions, no hidden model claims. Usually CPU and offline.      |
| B: Real model       | Show how the concept is used with an actual library/model | Reproducible environment, pinned model/data revisions, real inference or training, recorded outputs, explanation of API arguments. |
| C: Applied workflow | Solve and evaluate a realistic problem                    | Data preparation, baseline, suitable splits, evaluation, failure analysis, saved artifacts, and a transfer exercise.               |

These are not three versions of every page. Related lessons can share a level B/C lab with precise links to the relevant cells. Contextual lessons can use a decision activity instead of code. Every substantive method needs a level A demonstration or equivalent trace; every major book workflow needs levels B/C.

**A tiny arithmetic demo cannot be labeled a trained model. A script importing a library without using it does not count as an implementation. A full pipeline that never evaluates its output does not count as a finished applied workflow.**

### 5.2 Example package structure

Proposed structure, using one package per cohesive lab family rather than one environment per small snippet:

```text
examples/
  manifest.json
  foundations/
    pyproject.toml
    uv.lock
    README.md
    src/atlas_foundations/
    notebooks/
    tests/
  applications/
    pyproject.toml
    uv.lock
    README.md
    src/atlas_applications/
    notebooks/
    tests/
  training/
    README.md
    embedding/                 # separate lock when dependencies require it
    representation/
    generation/
  data/
    README.md
    fixtures/                  # small original or appropriately licensed data
    manifests/                 # provenance, revisions, hashes, split definitions
  verification/
    records/                   # compact run metadata and metric summaries
src/content/docs/labs/
src/data/book-sections.json
src/data/review-records.json
scripts/check-coverage.mjs
scripts/check-example-manifest.mjs
scripts/run-examples.py
tests/examples/
```

Choose and verify exact Python/package versions during E1. Do not invent a compatible lockfile in a plan or pin an entire ecosystem to arbitrary latest versions. Use version-controlled environments; split GPU-specific dependencies where CPU installations would otherwise fail.

Use reusable Python source as the implementation authority. Lesson snippets should be extracted from named source regions, and notebook code should import that source or be generated from it. Avoid three independently maintained copies of the same algorithm. Preserve explanatory notebook Markdown and interactive cells.

### 5.3 Manifest contract

Every executable example must declare:

| Field                                                  | Purpose                                                                                         |
| ------------------------------------------------------ | ----------------------------------------------------------------------------------------------- |
| Stable ID, title, concepts, source sections            | Traceability from book to lesson to executable work.                                            |
| Level and implementation kind                          | Distinguish a calculation, simulation, pretrained inference, fine-tuning, and applied workflow. |
| Entrypoint, environment, working directory             | Exact reproducible command and dependencies.                                                    |
| Model and tokenizer IDs plus revisions                 | Avoid silently downloading a different artifact.                                                |
| Dataset provenance, revision, split, license reference | Explain where data came from and prevent leakage.                                               |
| Seed and reproducibility limitations                   | Record controlled randomness and remaining nondeterminism.                                      |
| Runtime profile                                        | CPU/GPU/backend, RAM/VRAM, download size, disk, and measured duration when verified.            |
| Network/auth requirements                              | Distinguish first download, gated access, optional API calls, and fully offline execution.      |
| Input/output schema                                    | Describe expected shapes, dtypes, units, files, and result meaning.                             |
| Validation method                                      | Exact value, numerical tolerance, schema/invariant, or held-out metric.                         |
| Verification record                                    | Commit, command, date, environment, result, and evidence; absent until actually run.            |
| Troubleshooting and fallback                           | Supported alternatives and exact limitations of a smaller run or recorded result.               |

An expensive configuration may be documented before verification, but must remain visibly unverified. Never put fabricated runtime, memory, benchmark, or accuracy numbers into these fields.

### 5.4 The ideal notebook sequence

1. **Question and outcome:** State the task, what the learner will build, and what success means.
2. **Execution contract:** State environment, hardware, downloads, approximate measured duration, and any optional paid calls before execution.
3. **Inspect the data:** Show records, schema, class distribution or relevant corpus statistics, and provenance.
4. **Protect the evaluation:** Establish train/validation/test separation and deduplication before fitting, augmentation, or retrieval tuning.
5. **Run a baseline:** Use a simple meaningful method and explain its limitations.
6. **Build the method:** Explain each stage before its code; display intermediate arrays, shapes, tokens, masks, rankings, or gradients.
7. **Train or infer:** Explain essential settings and what actually changes. Show real outputs.
8. **Evaluate:** Compare the baseline and method on the same held-out data with the right metrics.
9. **Inspect failures:** Display representative errors and diagnose their likely cause.
10. **Ablate one choice:** Change one meaningful parameter or component; keep other conditions fixed.
11. **Save and reload:** Reproduce an inference result from the saved artifact, including tokenizer/config where required.
12. **Transfer:** Set a related exercise with hints, a solution, and a short report template.

Every notebook must pass **Restart kernel → Run all**. No out-of-order variables, hidden local files, missing imports, undisclosed manual edits, or stale outputs from a different model.

### 5.5 Code readability and correctness

- Prefer explicit, short functions with descriptive names and docstrings for nonobvious assumptions. Avoid clever one-liners that hide the algorithm.
- Add type hints for reusable interfaces; show tensor shapes at important boundaries.
- Explain _why_ a hyperparameter or transformation is chosen, not merely what its name is.
- Define inputs and fixtures before use. Use relative paths resolved from the lab, not author-specific paths.
- Validate empty inputs, malformed records, label mappings, truncation, padding, and unsupported devices where relevant.
- Test real behavioral properties, not only that a script finishes. For example, causal masking should eliminate future-position contribution, and frozen parameters should remain unchanged.
- Use tolerances for floating-point comparisons. Generated text is not a reliable exact-string test across devices and versions.
- Treat a short smoke run as an execution check, not proof of useful training. Report a full learning experiment separately.
- Do not force every experiment to improve. An honest negative result with a diagnosis is educational; never select the test set or seed to manufacture success.
- Include important configuration and exception handling without burying the mechanism in infrastructure.

### 5.6 Data and evaluation rules

- Provide small original fixtures so the first example is accessible without accounts or large downloads.
- Use appropriately licensed real datasets for realistic workflows; record source and retrieval scripts instead of silently redistributing data.
- Split by document, user, source, time, or entity when random row splitting would leak information. Explain the choice.
- Fit preprocessing and learned transformations only on the permitted training split. Isolate validation used for tuning from final test reporting.
- For augmentation, never create teacher labels from held-out examples and then train on them.
- Use the same evaluation cases across baseline and enhanced methods. Report denominators, sample sizes, and important slices.
- Explain micro/macro averages, threshold selection, ranking cutoffs, and span-level versus token-level metrics at their point of use.
- A saved artifact must include enough metadata to connect it to its model, data, configuration, and evaluation.

### 5.7 Runtime profiles and static hosting

| Profile             | Role                                                             | Promise                                                                                           |
| ------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Browser             | Small visual simulations                                         | No account; no server-side secrets; interactive calculation or clearly labeled recorded data.     |
| CPU                 | Foundational computations and small inference/training exercises | Verified clean local environment; published measured resource requirements.                       |
| GPU                 | Realistic fine-tuning and larger multimodal workflows            | Verified supported hardware/backend; explicit batch/sequence limits and recovery instructions.    |
| Optional hosted API | Recreate API-based classification/grounded-generation objectives | Explicit provider setup, bounded requests, usage visibility, and a local or recorded alternative. |

Do not promise CUDA-specific quantization on macOS/CPU. Offer an honestly labeled full-precision small-model alternative where feasible. Do not silently substitute random/tiny models and retain claims about realistic model quality.

GitHub Pages serves lessons, source downloads, notebooks, and precomputed result visualizations. Python training runs locally or in a notebook runtime. Public client JavaScript must never contain credentials. Hosted API examples are optional and are not invoked by site visits or ordinary CI.

### 5.8 Verification and CI design

- **Every content PR:** schema, source mapping, snippet synchronization, link/anchor checks, graph tests, changed level A examples, and the site release build.
- **Changed model lab:** clean-environment smoke run with a small configuration; validate tensor/data contracts, saving, reloading, and metric calculation.
- **Training or dependency change:** run the affected full supported experiment and attach an execution record before claiming that configuration is verified.
- **GPU/API jobs:** opt-in controlled execution; use recorded results for public browsing. Do not assume free runners provide a GPU or allow arbitrary paid calls.
- **Network failure:** distinguish an unavailable download from algorithm failure; retain the failure evidence and avoid a false verification badge.
- **PR trust:** never expose provider secrets to untrusted fork workflows or execute contributor code in a privileged deployment job.

Preserve `npm run release:build` as the site gate. Introduce separate documented commands for coverage and example verification, and add them only after their implementations exist. Keep heavy model tests independent of the normal Pages build.

## 6. Interactive lesson, graph, and search experience

### 6.1 Lesson layout

Preserve the existing visual identity and readable article column. Add a compact opening panel with the problem, objectives, prerequisites, and time/resource expectations. Use stable section links for **Explain → Trace → Experiment → Implement → Evaluate → Practice**; all content must remain readable without operating tabs.

Proposed reusable components:

- `LearningObjectives.astro`: concrete abilities and prerequisite links.
- `MathStep.astro`: formula, symbol definitions, dimensions, and numerical substitution.
- `WorkedTrace.astro`: input/intermediate/output table with code-region links.
- `ExperimentPanel.tsx`: predict, change a control, inspect, explain, reset.
- `CodeExample.astro`: synchronized source, copy/download, environment, expected result, verification record.
- `LabRunCard.astro`: notebook/local entry points, runtime requirements, and result status.
- `Misconception.astro`: incorrect belief, counterexample, corrected explanation.
- `Exercise.astro`: prompt, progressive hints, answer, and solution rationale.
- `CoverageStatus.astro`: source mapping and verified capabilities without implying mastery.

Do not build an elaborate component system before the three reference lessons show what is actually needed.

### 6.2 Rules for useful simulations

Every simulation needs a question, a prediction prompt, one or more meaningful controls, inspectable intermediate values, reset, and a text equivalent. Label the data source as **calculated**, **invented illustration**, or **recorded model output**.

Keep shared mathematical logic in testable modules, separate from React presentation. Reuse the same small fixtures in the Python explanation and browser simulation where possible; use explicit tolerances for cross-language comparisons.

Prioritize these mechanisms:

| Simulation             | Manipulation                                   | What the learner must observe                                                      |
| ---------------------- | ---------------------------------------------- | ---------------------------------------------------------------------------------- |
| Tokenizer comparison   | Text, tokenizer, special-token option          | Different splits and IDs; tokenizer-specific vocabulary; normalization effects.    |
| Attention trace        | Query/key/value values and causal mask         | Score matrix → scaled scores → masked softmax → weighted values.                   |
| Decoding               | Temperature, top-k, top-p, seed                | Distribution changes versus sampled token; deterministic greedy mode.              |
| KV cache               | Prompt length, generated length, head settings | Reused versus new computation and memory; not an invented hardware benchmark.      |
| Topic pipeline         | Representation/reduction/clustering choice     | Changed assignments versus changed topic descriptions, including outliers.         |
| Prompt experiments     | Examples, formatting, generation settings      | Tested task quality and validation outcomes; no universal prompt ranking.          |
| Tree search            | Branch limit, depth, evaluator                 | Generated candidates, pruned branches, cost, and a possible wrong selection.       |
| Retrieval              | Chunking, k, reranker, query rewrite           | Relevant document movement, context selection, and retrieval metrics.              |
| Multimodal preparation | Crop/resize and text prompt                    | The actual model input can differ from what the reader sees in the original image. |
| Contrastive learning   | Positive/negative pairs, temperature           | Similarity matrix, loss, and false-negative effects.                               |
| NER alignment          | Words, subword splits, tag policy              | Word IDs, label mask, decoded spans, and boundary errors.                          |
| LoRA/QLoRA             | Rank, dimensions, scale, precision             | Trainable count, update, quantization approximation, and storage estimates.        |
| Preference learning    | Chosen/rejected scores, reference, beta        | Relative preference objective; training loss versus external quality.              |

For model outputs that cannot run in the browser, show a precomputed experiment with its provenance and notebook link. A control must not claim to run a model when it only selects a stored result.

### 6.3 Graph enhancements

- Preserve the current **contains**, **prerequisite**, and **related** relationships and their validation.
- Add human-readable reasons: “requires embeddings because queries and documents are compared as vectors.”
- Separate **book location** from **conceptual parent**. A concept may be used in several chapters while keeping one canonical page and one display parent.
- Add optional relationship semantics such as **implements**, **evaluates**, **contrasts with**, and **used by** only after defining their direction and validation. Start with textual relationship details if extra edges would clutter the graph.
- Keep chapter overview and focused-neighborhood modes. Reveal cross-links on demand instead of rendering every edge at once.
- Show a “learn prerequisites” route and a “use this in a project” route from each concept.
- Integrate labs through inspector links initially; a later opt-in lab layer can add nodes without overwhelming the core concept map.
- Support aliases in graph search, stable shareable selection URLs, keyboard-accessible relationship lists, and preservation of selection through filter changes.
- Do not equate a parent with a prerequisite automatically. Audit actual learning dependencies and keep prerequisite edges acyclic.

### 6.4 Search and glossary

- Retain Pagefind for static full-text search; add canonical aliases and descriptive summaries to indexed content.
- Test acronym and full-name pairs: `NER`, `named entity recognition`, `SFT`, `instruction tuning`, `c-TF-IDF`, `LoRA`, `low rank adaptation`, `CoT`, and `tree of thought`.
- Build a glossary that links terms to their canonical definition and then to the full lesson.
- Add chapter, difficulty, and content-type filters if they improve search usability without complicating the default experience.
- Weight titles/definitions appropriately using documented Pagefind capabilities; exclude repeated navigation and boilerplate from results.
- Build a version-controlled search query set with expected destinations, including common confusions and use-case queries such as “train with a few labels.”
- Present helpful no-result suggestions from known aliases and nearby topics. Do not claim general typo correction or semantic search without implementing and verifying it.

### 6.5 Accessibility and performance acceptance

- Verify keyboard interaction, visible focus, meaningful labels, non-color distinctions, reduced motion, and text equivalents for each new visual.
- Check narrow screens, light/dark themes, zoomed text, long equations, code blocks, and graph inspector overflow.
- Keep controls close to the result they affect. Announce important changes without flooding assistive technology on every animation frame.
- Lazy-load interactive islands. Reading an ordinary lesson must not download model weights or the entire graph implementation.
- Record production-build payload and interaction baselines before setting regression thresholds. Test the enlarged graph with the actual final inventory.
- Manual usability checks supplement automation. A passing accessibility scanner alone does not establish that a complex simulation is understandable.

## 7. Chapter-by-chapter implementation backlog

The printed page ranges below identify the book's chapter scope; they are not PDF viewer page numbers. Existing lesson filenames are examples to revise, not an exhaustive list. E0 determines whether each additional concept needs a standalone page or a clearly anchored subsection. Proposed lab IDs `L01`–`L23` are stable work-package identifiers; a package may contain several short notebooks.

### 7.1 Chapter 1: introduction and model access — printed pages 3–34

**Learning outcome:** Explain the major model families, choose an appropriate access method for a task, and generate text while recognizing the limits of the result.

**Teaching work**

- Improve the historical progression from bag-of-words and dense vectors through attention, encoder/decoder models, and generative applications. Explain what problem each development addressed.
- Distinguish representation models, generative models, and encoder-decoder systems; connect architecture to useful tasks rather than brand names.
- Explain the changing meaning of “large,” pretraining versus later adaptation, and the historical context of the book's generative-AI discussion. Label dated claims.
- Add model access: hosted/proprietary models, downloadable/open-weight models, licensing considerations, framework roles, and local versus remote execution.
- Clarify that downloadable weights and fully open development are different properties; compare access, control, data handling, operating requirements, and reproducibility.
- Add a limited-resource learning route and a first-generation walkthrough. Explain model/tokenizer pairing, prompt input, continuation, and decoding.
- Expand responsible use with concrete error checking, privacy-aware input choice, task boundaries, and source verification examples.

**L01: first text generation and access choices**

Build a small local inference notebook with environment inspection, a verified accessible model, fixed prompts, encoded input, generation configuration, and decoded continuation. Display the distinction between the prompt and newly generated text. Include a historical/API access explanation with an optional provider implementation; the default lesson remains usable without a key.

**Visual and practice:** An architecture-to-task map and an access decision table. Ask learners to select a model family/access pattern for document classification, offline drafting, and a privacy-sensitive task, then justify the tradeoffs.

**Acceptance:** A clean run produces real generation and explains its variability. The learner can identify which work happens locally or remotely and why a fluent answer is not evidence of correctness. Starting files include `language-ai.mdx`, `llm.mdx`, model-family lessons, and `responsible-use.mdx`.

### 7.2 Chapter 2: tokens and embeddings — printed pages 37–71

**Learning outcome:** Trace text into tokenizer IDs and vector representations, compare representations, and train a small recommendation embedding model.

**Teaching work**

- Expand word, subword, character, and byte tokenization; vocabulary, merges where applicable, unknown tokens, special tokens, whitespace, casing, normalization, and tokenizer-specific IDs.
- Compare trained tokenizers on the same English text, Korean or another non-English script, code, punctuation, emoji, and unusual words. Distinguish token count from word count and from model quality.
- Show tokenizer output, attention mask, embedding lookup shape, contextual hidden states, and sentence/document pooling. Explain padding and truncation.
- Explain why contextual vectors for the same word can differ and why a contextual token vector is not automatically a good sentence embedding.
- Show actual pretrained word embeddings, cosine similarity, and nearest-neighbor inspection, including a misleading similarity case.
- Expand Word2vec and contrastive training: context windows, positive pairs, negative sampling, learned parameters, and the training objective.
- Add the missing song-recommendation workflow: playlists as sequences, songs as IDs, co-occurrence as signal, training, retrieval, cold-start limits, and popularity bias.

**L02: tokenizer comparison:** Print token strings, IDs, offsets when supported, special-token masks, and round-trip behavior. Record tokenizer revisions. Test empty text, multilingual inputs, and normalization-aware round trips without assuming every tokenizer preserves every byte.

**L03: representations:** Compare static word vectors, contextual token states, and sentence embeddings. Display dimensions and a cosine matrix; use a sentence with an ambiguous word and a contrasting context. Label any 2D projection separately from the original vectors.

**L04: playlist embeddings:** Create a small original playlist fixture, train embeddings from co-occurrences, and recommend unseen items. Add a larger permitted dataset option. Compare against a popularity baseline using a leakage-aware split and ranking metric; exclude already observed items where the task requires it.

**Visual and practice:** Token-to-ID-to-vector trace; vector projection with limitations; context-window pair builder; recommendation neighborhood. Exercises include predicting tokenizer differences, computing cosine similarity, and diagnosing a cold-start item.

**Acceptance:** Recommendation vectors are learned, not hardcoded. Test playlist separation and held-out targets. No success claim rests on retrieving the input song itself. Update `tokenization.mdx`, embedding lessons, `word2vec.mdx`, and `recommendation-embeddings.mdx`.

### 7.3 Chapter 3: transformer internals — printed pages 73–106

**Learning outcome:** Trace a transformer forward pass, explain decoding and caching, and relate architectural changes to computation and memory.

**Teaching work**

- Show tokenizer input, embedding lookup, transformer blocks, LM head, logits, probabilities, selected token, and autoregressive repetition.
- Explain parallel processing of an available sequence versus sequential generation of new tokens; distinguish training, prompt processing, and decoding.
- Expand scaled dot-product attention with Q/K/V shapes, masking, softmax, weighted values, heads, and output projection.
- Teach residual paths, feedforward layers, activation functions, LayerNorm versus RMSNorm, and pre-normalization as the book's architectural variants.
- Expand position handling: absolute positions, RoPE rotations of queries/keys, and implications for relative position. Explain packing and document boundaries without implying packing is automatically leakage-free.
- Compare multi-head, multi-query, and grouped-query attention; explain cache size tradeoffs. Explain FlashAttention as an implementation strategy for exact attention, distinct from changing which positions attend.
- Cover sliding/local attention where discussed and the book's architectural research directions. Mark later additions as extensions.

**L05: transparent transformer:** Implement a tiny forward pass with explicit shapes and a causal mask, then compare selected computations to a trusted tensor-library reference. Add a real small-model inspection notebook. Random weights teach mechanics; they do not demonstrate learned language ability.

**L06: decoding and KV cache:** Implement stable softmax, temperature, top-k/top-p, seeded sampling, and greedy decoding. Compare cached and uncached logits for the same supported model and inputs within tolerance. Separate theoretical operation/memory estimates from measured timings, with device synchronization and warm-up when benchmarking.

**Visual and practice:** Step through an attention matrix; rotate a vector pair for RoPE; inspect cache growth. Exercises ask learners to find an incorrect mask, explain why larger temperature changes diversity, and calculate a cache shape.

**Acceptance:** Future positions receive zero attention in the masked example; shapes match equations; cached/uncached correctness is checked before speed. Existing attention/transformer/decoding/cache/efficient-attention lessons all receive a substantive review.

### 7.4 Chapter 4: text classification — printed pages 111–135

**Learning outcome:** Choose, implement, and fairly compare classification methods under different labeling and compute constraints.

**Teaching work**

- Expand movie-review sentiment as a concrete task, dataset inspection, split choice, label mapping, and model selection.
- Teach task-specific classification, embeddings plus a classifier, supervised classification, and classification without labeled examples.
- Distinguish label-description similarity, entailment-style zero-shot classification, and generative classification; explain what each score means.
- Add the book's T5/text-to-text approach with encoder-decoder mechanics and label generation.
- Add the hosted chat-classification workflow represented by the book's ChatGPT example, with current implementation details checked at implementation time.
- Explain invalid labels, ambiguous cases, thresholds/abstention, class imbalance, batching, and data contamination concerns.

**L07: classification comparison:** Use one versioned dataset and identical held-out cases across a simple baseline, task-specific encoder, embedding classifier, zero-shot method, and text-to-text model. Put an optional hosted-model comparison in a separate notebook. Report macro-F1, per-class precision/recall, confusion matrix, invalid-output rate, and measured operational characteristics where relevant.

**Visual and practice:** Decision boundary/threshold view, confusion matrix, and side-by-side method errors. Include negation, mixed sentiment, and domain-shift examples; ask learners to choose a method when only a few labels exist.

**Acceptance:** Label IDs and textual outputs map consistently; no tuning on test cases; the notebook includes at least one error analysis and explains why methods disagree. Upgrade the classification concepts and the existing classification use case together.

### 7.5 Chapter 5: clustering and topic modeling — printed pages 137–164

**Learning outcome:** Build a real topic pipeline and distinguish changes to document groups from changes to their descriptions.

**Teaching work**

- Introduce the research-abstract corpus, provenance, preprocessing, and what an unlabeled task can and cannot establish.
- Expand document embedding, UMAP/dimensionality reduction, HDBSCAN/density clustering, outliers, cluster inspection, and projection limitations.
- Teach the difference between ordinary TF-IDF and class-based TF-IDF using a hand-worked table with defined quantities.
- Explain every replaceable BERTopic stage and what information passes between stages.
- Add the missing representation modules: KeyBERTInspired, maximal marginal relevance/diversity, and generation-based topic descriptions. Distinguish representation refinement from model weight fine-tuning.
- Explain topic visualization, representative documents, stability, coherence limitations, and human labeling decisions.

**L08: modular BERTopic:** Fit a genuine BERTopic pipeline on a small reproducible corpus, inspect outlier labels, display representative documents, and save/reload the result. Compare default representations, keyword refinement, diversity-based refinement, and an optional local text-generation representation. Use the same clustering assignments when comparing representation-only changes. Check the [official representation documentation](https://maartengr.github.io/BERTopic/getting_started/representation/representation.html) against the chosen version.

**Visual and practice:** Pipeline stage inspector, c-TF-IDF arithmetic, and before/after topic descriptions. Ask learners to find an attractive but unsupported generated label and explain why a 2D grouping may mislead.

**Acceptance:** Code actually computes the pipeline; fixed cluster labels are confined to the introductory arithmetic example. Include meaningful human inspection, a changed-component experiment, and an explanation of label instability. Revise `bertopic.mdx`, `topic-representations.mdx`, and connected clustering lessons.

### 7.6 Chapter 6: prompting and structured output — printed pages 167–198

**Learning outcome:** Design and evaluate prompts, distinguish prompting strategies, and validate structured output.

**Teaching work**

- Cover model selection/loading, chat versus base-model formatting, output controls, prompt ingredients, role/instruction/context/example distinctions, and increasing prompt complexity.
- Deepen instruction prompting, in-context learning, few-shot example selection/order, and chain prompting.
- Add standalone explanations for **chain-of-thought** and **tree-of-thought**. Distinguish a visible worked solution, a multi-call workflow, candidate search, and hidden internal computation.
- Expand self-consistency: sampling, answer extraction, vote aggregation, tied/invalid answers, and added compute. Repeated samples are not independent guarantees.
- Teach output verification, demonstrations of desired format, schema validation, and constrained decoding. Formatting instructions alone are not grammar enforcement.

**L09: prompt evaluation:** Define a small held-out task suite with verifiable answers and structured extraction targets. Compare prompt variants under controlled model/generation settings. Report answer accuracy, format validity, and work/cost measurements. Keep prompt-design cases separate from evaluation cases.

**L10: explicit search and verification:** Implement a small tree search over a solvable task, showing state, candidate expansion, evaluator, pruning, and stopping. Add a model-based candidate variant if supported. Include self-consistency and an actual supported constrained-decoding example; otherwise label the unsupported backend and keep it incomplete rather than substituting a regex repair while claiming constrained sampling.

**Visual and practice:** Prompt assembly, candidate voting, branch exploration, and validator failures. Ask learners to diagnose a persuasive but incorrect intermediate step and a syntactically valid answer with incorrect content.

**Acceptance:** Prompt strategies are compared with task evidence, not a single pleasing output. The tree contains real candidate evaluation and branching. The lesson states that visible reasoning text is not proof of faithful internal reasoning. Add new concept IDs after the E0 deduplication review.

### 7.7 Chapter 7: chains, memory, and agents — printed pages 199–224

**Learning outcome:** Assemble and inspect a controlled language-model workflow and explain the state and tool boundaries.

**Teaching work**

- Add the missing LangChain implementation layer while preserving a framework-independent explanation of each mechanism.
- Explain quantized local model loading, model I/O adapters, prompt templates, a single chain, and composition of multiple prompts.
- Teach conversation buffers, windowed memory, summarization memory, token budgets, and the loss of information caused by truncation or summarization.
- Distinguish an application retaining conversation data from a model changing its weights.
- Explain agents, tool schemas, ReAct-style iteration, observations, stop conditions, invalid calls, and useful limitations.
- Compare the book's historical API to the chosen supported implementation; do not copy obsolete imports as if verified today. Consult the [official LangChain documentation](https://docs.langchain.com/oss/python/langchain/overview).

**L11: workflow toolkit:** Provide short notebooks for (a) local model adapter and prompt chain, (b) three memory policies on the same conversation, and (c) a bounded tool-using loop. Show a minimal plain-Python version before the framework equivalent. Use deterministic calculator/fixture-search tools with no arbitrary code execution, private-data access, or write side effects. Include a real model path and a separately labeled fake-model test path.

**Visual and practice:** Workflow state trace showing prompt inputs, messages, tool arguments/results, retained memory, and stop reason. Exercises diagnose missing conversation facts and loops that should terminate.

**Acceptance:** A clean supported environment runs the framework workflow; token/memory accounting is inspectable; invalid tool arguments and max-step termination are tested. Rework `chains.mdx`, memory lessons, `agents.mdx`, and `react.mdx` with explicit reasons for their relationships.

### 7.8 Chapter 8: retrieval and RAG — printed pages 225–258

**Learning outcome:** Build and evaluate retrieval and grounded generation as separate stages, then justify advanced retrieval choices.

**Teaching work**

- Deepen corpus preparation, query/document embeddings, dense retrieval, similarity search, candidate selection, cross-encoder reranking, and ranking metrics.
- Cover the book's chunking discussion (printed pages 235–238), including overlapping passages and retained source metadata; mark further implementation extensions explicitly.
- Work through the book's precision at k, average precision (AP), mean average precision (MAP), and nDCG discussion with small ranked lists and explicit relevance labels. Recall at k and MRR can provide supplementary comparisons. The E0 audit identified AP/MAP as missing from the earlier plan.
- Add full local-model and API-based grounded-generation workflows with retrieved context and attributable claims.
- Cover the book's advanced techniques: **query rewriting, multi-query retrieval, multi-hop retrieval, source routing, and agentic retrieval**. Explain their different information flows.
- Separate retrieval relevance, answer correctness, faithfulness, citation precision/recall, perceived usefulness, and fluency. Teach when human assessment remains necessary.
- Introduce any hybrid lexical/dense retrieval comparison as an optional extension, not an invented book subsection.

**L12: retrieval benchmark:** Build a small versioned corpus and held-out query/relevance set. Compare a lexical baseline, dense retrieval, and reranking. Preserve source/chunk IDs; display top-k rankings and metric calculations. Include queries whose answers are absent.

**L13: grounded QA and advanced retrieval:** Add real local generation, an optional hosted adapter, context construction, citations, and abstention. Compare rewriting, multi-query, multi-hop, and routing on purpose-built cases; keep the baseline visible and account for additional calls. Add a human rubric and an optional automated judge with limitations. Treat retrieved text as source data rather than executable instructions.

**Visual and practice:** Follow one query through all stages and show which stage caused an error. Ask learners to repair a query rewrite that changed the user's intent and a citation that does not support its attached claim.

**Acceptance:** Citation IDs resolve, absent-evidence cases are evaluated, retrieval and generation scores are reported separately, and advanced variants are not assumed to improve every case. Upgrade the existing RAG use case and learning path alongside the concepts.

### 7.9 Chapter 9: multimodal models — printed pages 259–286

**Learning outcome:** Explain image/text representations and run image retrieval, captioning, and multimodal prompting with the actual model inputs visible.

**Teaching work**

- Deepen vision transformer patching, positional information, and image tensor shapes.
- Explain CLIP's paired training objective and the distinction between shared embedding comparison and text generation.
- Add a real OpenCLIP workflow with both encoders, normalization, similarity, and prompt sensitivity.
- Expand BLIP-2's frozen components, bridging/query mechanism, and generation pipeline.
- Add the missing preprocessing walkthrough: image decoding, color conversion, resize/crop, normalization, processor outputs, text tokenization, and device/dtype movement.
- Distinguish captioning from visual question answering and chat formatting. Show crop-related information loss, counting failures, and visually unsupported text.

**L14: multimodal suite:** Use a small original or clearly licensed image set. Include OpenCLIP image/text retrieval, a genuine BLIP-2 captioning configuration, and a supported multimodal chat example. Display original and processed images, tensor shapes, prompts, ranked similarities, and real generated outputs. Separate CPU-friendly retrieval from GPU-heavy generation when necessary.

**Visual and practice:** Image-to-patches trace, original/cropped comparison, and image/text similarity matrix. Ask learners to predict how cropping changes a question's answerability and to identify a caption detail unsupported by the image.

**Acceptance:** Emoji/symbolic vectors remain labeled illustrations; model labs operate on real image pixels. Each sublab has verified preprocessing and a declared environment. Recorded output is available for learners unable to run the larger model.

### 7.10 Chapter 10: embedding model training — printed pages 289–321

**Learning outcome:** Construct suitable training examples, select an embedding objective, and evaluate adaptation without contaminating evaluation data.

**Teaching work**

- Expand SBERT/bi-encoder architecture, pooling, pairwise comparisons, and the role of contrastive examples.
- Work through the book's softmax/classification, cosine-similarity regression, and multiple-negatives ranking objectives. Define inputs, labels, logits/scores, reduction, and gradients at an appropriate depth.
- Explain positive pairs, explicit negatives, in-batch negatives, hard negatives, false negatives, and batch composition. Avoid treating related contrastive losses as identical in every implementation.
- Add the full data → objective → trainer → evaluator → saved model lifecycle and deeper embedding evaluation.
- Add **Augmented SBERT**: gold labeled pairs → cross-encoder teacher → candidate pairs → silver labels → bi-encoder student. Explain that silver labels may be wrong.
- Expand supervised fine-tuning, unsupervised TSDAE reconstruction, corruption, and TSDAE-based domain adaptation.

**L15: embedding training and losses:** Start with hand calculations and tensor assertions; train a small real embedding model with appropriate paired data and compare against its unmodified baseline. Include a matching evaluator and a retrieval or similarity task. Check the [Sentence Transformers training guide](https://www.sbert.net/docs/sentence_transformer/training_overview.html) against the selected versions.

**L16: Augmented SBERT:** Train or explicitly load a documented teacher, generate silver labels from training-pool pairs only, train gold-only and gold-plus-silver students, and compare on untouched gold evaluation data. The full book workflow should include teacher training; an inference-only teacher shortcut must be labeled partial.

**L17: TSDAE and adaptation:** Corrupt sentences, reconstruct the original text, explain the architecture and objective, and evaluate adapted embeddings on a downstream held-out task. Include a reconstruction example and a case where lower reconstruction loss does not improve retrieval.

**Visual and practice:** Similarity matrix/loss explorer, teacher-student data flow, and corruption/reconstruction trace. Exercises include finding a false negative and detecting silver-label leakage.

**Acceptance:** Training changes the intended parameters; evaluators match the objective; saved embeddings can be reloaded; gold/silver provenance is explicit; results include a baseline and honest interpretation.

### 7.11 Chapter 11: representation-model fine-tuning — printed pages 323–353

**Learning outcome:** Fine-tune and compare representation models for classification and NER, including low-label and domain-adaptation settings.

**Teaching work**

- Deepen pretrained BERT plus task head, tokenization, batches, optimizer settings, validation, and model selection.
- Explain frozen layers, trainable parameters, gradient flow, and tradeoffs in data and compute.
- Add a real SetFit workflow and explain its embedding adaptation and classifier stages; evaluate across a documented few-shot sampling protocol.
- Expand continued pretraining with masked language modeling, random masking, ignored labels, domain corpus selection, and downstream evaluation.
- Complete NER: annotation guidelines, BIO tagging, word/subword mapping, ignored special/padding tokens, model head, loss, decoding, and entity-level metrics.
- Explain invalid BIO sequences, truncation, nested entities outside a simple flat tagger's scope, and domain-specific boundary conventions.

**L18: BERT and SetFit classification:** Compare full/partial freezing and a few-shot SetFit route against the earlier classification baseline. Reuse splits where appropriate, report per-class outcomes, and repeat few-shot selection enough to expose variability rather than reporting only one favorable sample.

**L19: continued pretraining:** Perform real masked-language-model adaptation on a small permitted domain corpus, then compare downstream classification before and after. Separate a quick execution smoke configuration from the evaluated adaptation run.

**L20: complete NER:** Prepare labeled records, verify subword label alignment, fine-tune a token-classification model, decode entities, save/reload, and report entity-level precision/recall/F1. Follow the model/tokenizer-specific requirements in the [official token-classification guide](https://huggingface.co/docs/transformers/tasks/token_classification).

**Visual and practice:** Editable word/subword/BIO alignment table and span-error viewer. Ask learners to correct a boundary error that token accuracy conceals and to identify which subwords contribute to loss.

**Acceptance:** Alignment tests include split words, special tokens, padding, truncation, and multiword entities. Frozen parameters actually remain fixed. Test data does not enter continued pretraining. All classification and NER examples show real predictions rather than supplied labels masquerading as model output.

### 7.12 Chapter 12: generation-model fine-tuning — printed pages 355–389

**Learning outcome:** Prepare instruction and preference data, adapt a generation model, verify its artifacts, and evaluate the resulting behavior.

**Teaching work**

- Distinguish pretraining, supervised instruction tuning, and preference tuning by data, objective, updated parameters, and expected behavioral change.
- Deepen full fine-tuning versus PEFT; quantify trainable parameters while separating weights, activations, gradients, optimizer state, and KV cache.
- Explain LoRA dimensions, rank, scaling, target modules, initialization, dropout where used, and the frozen base. Explain QLoRA's quantized base separately from adapter precision.
- Add instruction-data templating, role boundaries, special tokens, EOS, truncation, padding, labels, and assistant/completion-only loss when applicable. Check masks rather than assuming a trainer default matches the lesson.
- Add training configuration: batch size, accumulation, sequence length, learning rate/schedule, epochs, precision, checkpointing, validation, and save/reload.
- Add adapter merging with supported precision/quantization paths, tokenizer preservation, output equivalence checks, and memory implications.
- Expand evaluation: perplexity, lexical overlap metrics, semantic similarity metrics such as BERTScore, benchmarks/leaderboards, automated judges, and human assessment. Explain what each measures and fails to measure rather than grouping them all as word-overlap metrics.
- Deepen preference data, reward model inputs/outputs, pairwise reward training, the role of policy/reference models, conceptual RLHF, and DPO.
- Add preference-data templates, prompt/chosen/rejected separation, truncation controls, reference behavior, and complete DPO training/evaluation.

**L21: SFT, LoRA/QLoRA, and merging:** Use a small original instruction fixture for inspection/tests and a permitted realistic dataset for the evaluated run. Include a tiny full-fine-tuning comparison, a LoRA run, and a verified QLoRA configuration. Show trainable counts, before/after held-out behavior, artifact reload, and supported merging. Follow the selected versions of [chat templating](https://huggingface.co/docs/transformers/chat_templating), [PEFT LoRA](https://huggingface.co/docs/peft/developer_guides/lora), and [TRL SFT](https://huggingface.co/docs/trl/sft_trainer) documentation.

**L22: generation evaluation:** Create a fixed task suite with reference-based metrics where appropriate, objective checks, error categories, and a blinded human rubric. Add optional automated judging with prompt/version records, order controls, and calibration examples. Demonstrate a metric disagreement instead of declaring one universal score.

**L23: reward modeling and DPO:** Prepare preference pairs, train a small reward model, evaluate held-out ranking, then separately fine-tune a policy with DPO from a suitable SFT starting point. Explain that DPO does not require the separately trained reward model. Save/reload and compare base/SFT/DPO behavior with both preference metrics and task quality. Verify the chosen configuration against the [TRL DPO documentation](https://huggingface.co/docs/trl/dpo_trainer).

**Visual and practice:** Loss-mask inspector, rank/update explorer, training-memory breakdown, preference-pair trace, and evaluation disagreement dashboard. Exercises include diagnosing a duplicated chat token, wrong loss mask, invalid chosen/rejected pair, and a merged adapter loaded against the wrong base.

**Acceptance:** Data masks and reference usage are inspected; trainable parameters and artifacts are verified; merge equivalence uses appropriate tolerances; benchmark/test prompts do not become training data. If no supported GPU run is available, the affected workflow remains explicitly unverified and the full completion gate remains open.

## 8. Three precise examples of the required depth

These are implementation specifications, not claims that the examples have already been written or run.

### 8.1 LoRA: connect equations to an actual update

Define the orientation before writing code:

- `x` is a column vector with shape `(d_in, 1)`.
- `W` has shape `(d_out, d_in)`.
- `A` has shape `(r, d_in)`; `B` has shape `(d_out, r)`.
- For the standard scaling used in this example, `delta_W = (alpha / r) * B @ A` and `y = W @ x + (alpha / r) * B @ (A @ x)`.
- Identify variants separately; do not imply every LoRA implementation uses exactly this scaling.

Use the trace `W = [[1, 0], [0, 1]]`, `A = [[0.3, -0.1]]`, `B = [[0.1], [0.2]]`, `x = [[2], [1]]`, `r = 1`, `alpha = 1`.

The worked result is `A @ x = [[0.5]]`, `B @ (A @ x) = [[0.05], [0.10]]`, and `y = [[2.05], [1.10]]`. Show the same values in the equation, code, and visual. Explain that a 2-by-2 rank-one example has four adapter parameters and offers no parameter-count saving over its four base parameters; use larger dimensions to demonstrate the benefit.

Then extend to a tiny trainable task. Compare a frozen base, full update, and adapter update. Show the loss and inspect gradients. With the common initialization of random A and zero B, explain why the initial update is zero and why gradient behavior differs between A and B at the first step; do not zero both matrices and accidentally prevent learning.

Required checks: base weights stay fixed in the adapter run; the expected parameters receive gradients; factorized and merged outputs agree within tolerance with dropout disabled for comparison; save/reload preserves inference. The real-model lab must additionally cover target modules, environment, and held-out behavior.

### 8.2 NER: make the difficult alignment step visible

Start with an original annotated sentence containing both a multiword entity and a word that the selected tokenizer splits. Define the annotation policy before displaying labels.

Show a table with original word index, word, word-level BIO label, subtoken, token ID, tokenizer word ID, training label, and loss inclusion. Explain the chosen labeling policy: labeling only the first subtoken and masking the rest is different from propagating labels to all subtokens; propagation must handle `B-`/`I-` correctly.

The example must:

1. Use the actual tokenizer's mapping; do not guess splits or assume one word equals one token.
2. Mask special tokens, padding, and ignored subtokens using the training library's ignore convention.
3. Explain how truncation affects a partly retained entity and choose a documented policy.
4. Train a model and decode its predictions into typed spans.
5. Compare a correct entity, an incorrect type, and an incorrect boundary under strict span-level scoring.
6. Show why high token accuracy may coexist with poor entity extraction when most tokens are outside entities.

Required checks: token/label lengths agree; masks are correct; empty-entity sentences are supported; a multiword entity is reconstructed correctly; metric code is checked against a manually scored fixture.

### 8.3 BERTopic: distinguish assignments from descriptions

Start with a small original corpus containing three interpretable themes plus ambiguous/outlier documents. Explain the corpus's educational limitations.

Trace each document through its embedding, reduced representation, assigned cluster, and representative topic terms. Work one c-TF-IDF term calculation by hand using the same stated convention as the code; do not substitute ordinary within-group word frequency and label it c-TF-IDF.

Compare two controlled experiments:

1. **Representation-only change:** Keep document assignments fixed, then compare baseline c-TF-IDF terms, semantic keyword reranking, diversity adjustment, and generated labels.
2. **Clustering change:** Change one upstream clustering/reduction setting and inspect which documents move, become outliers, or alter topic interpretation.

Required checks: a genuine pipeline fits; topic assignments correspond to input rows; outliers are handled; representation-only changes preserve assignments; saved artifacts reload; generated labels are checked against representative documents. Explain that coherent-looking names are not proof that the clusters are stable or useful.

## 9. Applied projects and learning paths

Connect chapters through projects that produce inspectable artifacts. Reuse data and evaluation components where appropriate, but do not pretend a single dataset suits every task.

| Project                               | Concepts and labs           | Required deliverable                                                                        | Transfer question                                                        |
| ------------------------------------- | --------------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| Review or support-ticket classifier   | Chapter 4, L07, L18         | Baseline/model comparison, confusion matrix, error analysis, inference script               | What changes when the domain or class balance changes?                   |
| Research collection explorer          | Chapter 5, L08              | Inspectable topic pipeline, representative documents, label comparison, reproducible report | Does a changed topic name indicate changed document assignments?         |
| Grounded document assistant           | Chapters 7–8, L11–L13       | Retriever, generator, citations, absent-answer behavior, separate stage evaluation          | Is an error caused by missing evidence or unsupported generation?        |
| Image/text collection search          | Chapter 9, L14              | Real embeddings, retrieval interface/recorded results, preprocessing report                 | Which query/crop changes improve retrieval, and which lose information?  |
| Domain embedding adapter              | Chapters 10–11, L15–L17/L19 | Training data provenance, adapted model, held-out evaluation, baseline comparison           | Does better training loss transfer to the user's retrieval task?         |
| Entity extraction pipeline            | Chapter 11, L20             | Annotation policy, alignment code, trained tagger, decoded spans, entity metrics            | Which boundary conventions must change for a new domain?                 |
| Instruction and preference adaptation | Chapter 12, L21–L23         | Prepared datasets, adapter/model artifacts, before/after results, evaluation rubric         | Did the model become more useful, or only better at the training format? |

Update the existing three paths rather than discarding their URLs. Add focused routes as the corresponding labs become available:

- **Foundations:** text → tokens → embeddings → attention → decoding → cache.
- **Choosing an application approach:** task-specific models → embeddings → generative classification → prompting → evaluation.
- **Discovering structure:** document representations → reduction → clustering → topic descriptions → inspection.
- **Building grounded answers:** retrieval → reranking → context → generation → citations → RAG evaluation.
- **Adapting models:** data/splits → objectives → embedding or classifier training → instruction tuning → preferences → evaluation.
- **Understanding multimodal inputs:** image preparation → vision representation → shared embeddings → caption/chat generation.

Each path needs entry requirements, expected artifacts, checkpoints, a final transfer task, and a recovery link when the learner misses a prerequisite. Keep estimated learning time provisional until a reader has attempted the path.

### 9.1 Exercises and learner feedback

Use four question types where appropriate:

1. **Recall:** Explain the concept and its nearest confusing alternative.
2. **Predict/calculate:** Work a small case before running the visual or code.
3. **Debug:** Repair an incorrect mask, split, metric, prompt, label, or configuration.
4. **Transfer:** Apply the method to a new task and justify the evaluation choice.

Give progressive hints before a complete solution. Show why a plausible wrong answer is wrong. For subjective decisions, provide a rubric and alternative defensible answers instead of a fake single correct choice.

The existing local completion marker should continue to mean “marked complete by the reader.” Do not label it mastery. If richer practice progress is added, version the localStorage format, migrate existing completed IDs, provide reset/export, and distinguish **read**, **practiced**, and **self-assessed** states. Account-free reading remains the default.

For pilot feedback, ask readers to perform specific tasks and record where they become stuck. Request explanations in their own words, not only a satisfaction score. A sample of readers informs revisions; it does not prove every audience can understand every lesson.

## 10. Implementation phases, dependencies, and completion gates

Use enhancement phase IDs **E0–E10** to distinguish this work from the original Phases 1–5. Do not mark a whole enhancement phase complete because its Markdown exists. Each gate requires the artifacts and evidence below.

### E0 — establish the audited inventory

**Depends on:** No new implementation.

- [x] Review the supplied book's full detailed structure, important figures, embedded concepts, and official chapter notebooks.
- [x] Create `book-sections.json` with source locations, objectives, mappings, required dimensions, and unresolved gaps.
- [x] Audit all 96 current lessons; mark depth and implementation gaps without deleting useful content.
- [x] Record proposed new canonical concepts and aliases; decide page versus subsection versus shared lab.
- [x] Correct the broad coverage claims in README, plan, and the sources page.
- [x] Add inventory validation and a generated review report; preserve existing navigation data until migration is complete.

**Gate:** All 12 chapters have been audited, every required objective has a destination or explicit missing-work entry, and the report distinguishes source coverage from teaching depth. No arbitrary target lesson count is set in advance.

**Review artifact:** [Source inventory](./src/data/book-sections.json), [chapter gap report](./docs/audits/book-gap-report.md), [lesson depth report](./docs/audits/lesson-depth-report.md), [proposed destinations and aliases](./src/data/proposed-concepts.json), and [audit/migration notes](./docs/audits/README.md). All 96 lessons remain marked as requiring enhancement review; external model notebooks were inspected, not executed.

### E1 — build reproducible example infrastructure

**Depends on:** E0's identifiers and source mapping contract.

- [x] Implement the example manifest and schema; register existing snippets before replacing their checker.
- [x] Create a verified foundation environment and one supported application/training environment slice; document later environment families.
- [x] Add fixture provenance, data split manifests, artifact conventions, and execution-record schemas.
- [x] Implement source-region inclusion/synchronization for lesson snippets and notebook imports/generation.
- [x] Add example discovery, environment checks, timeouts, output/invariant validation, and verification report generation.
- [x] Replace exact 76-example assumptions with manifest completeness checks; do not silently stop checking the old examples.
- [x] Separate quick PR checks from model/GPU/API execution. Document every supported command in the relevant README.

**Gate:** One offline mechanism example and one small real-model example run from clean environments, produce traceable records, and display synchronized code/output in a lesson. Existing example coverage does not regress.

**Review artifact:** [Manifest](./examples/manifest.json), [environment and execution guide](./examples/README.md), [generated lab](./src/content/docs/labs/first-verified-examples.mdx), [verification report](./examples/verification/README.md), and committed run records. E1 verifies CPU inference and mechanisms; training environments follow in E2 and later phases.

### E2 — finish the three reference lessons and shared UI

**Depends on:** E0–E1.

- [x] Rewrite tokenization, LoRA, and NER using the teaching sequence and precise contracts in Sections 3 and 8.
- [x] Deliver the tokenizer comparison, tiny trainable LoRA mechanism plus a supported adapter run, and a complete small NER training workflow.
- [x] Build only the reusable lesson components these examples actually require.
- [x] Add equation/shape traces, topic-specific exercises, real error cases, and aligned simulation/code fixtures.
- [ ] Obtain independent editorial and technical review; verify a beginner can perform the main task without unexplained steps.
- [x] Document the resulting reference pattern in CONTRIBUTING and a lesson template.

**Gate:** All three lessons pass the rubric and their supported examples execute. L02/L20 and the initial LoRA portion of L21 establish reusable references; later phases extend dataset choices, QLoRA, merging, and broader evaluations rather than rewriting the same work.

**Review artifact:** [Three reference lesson author review](./docs/reviews/e2-reference-lessons.md), [locked environment and execution guide](./examples/training/reference/README.md), [lesson template](./docs/templates/reference-lesson.md), synchronized notebooks, and committed execution records. Implementation and author technical checks are delivered; independent review/beginner feedback remain pending, so the full E2 acceptance gate remains open.

### E3 — foundations and model access

**Depends on:** E2.

- [x] Complete Chapter 1–3 work from Section 7, including missing access/context material and prerequisite bridges.
- [x] Complete L01–L06, reusing the reference tokenizer work.
- [x] Upgrade attention, decoding, cache, and architecture visuals with tested quantities and clear assumptions.
- [x] Revise all affected existing foundation lessons, including generic worked examples and repeated self-checks.
- [x] Update the sentence-processing path, graph dependencies, glossary aliases, and chapter coverage views.

**Gate:** A learner can generate text, compare tokenizers, inspect representations, train a small recommendation model, and trace a transformer computation. Numerical properties and cached/uncached correctness are verified.


**Implementation evidence:** [Foundation labs](./src/content/docs/labs/foundations.mdx), [CPU guide](./examples/training/reference/E3.md), and [author review](./docs/reviews/e3-foundations.md). L02 reuses the E2 tokenizer package. L01/L03–L06 have recorded CPU runs. The optional larger Last.fm route is importer-tested, not a claimed full-data experiment. Independent technical/editorial and beginner review remain open; implementation does not award acceptance.

### E4 — classification and topic discovery

**Depends on:** E3; application environment from E1.

- [x] Complete Chapter 4–5 work, including T5 classification, optional hosted classification, and real BERTopic representation modules.
- [x] Complete L07–L08 with shared evaluation fixtures and honest baseline comparisons.
- [x] Upgrade the classification use case and add the research collection explorer project.
- [x] Add confusion-matrix, c-TF-IDF, cluster/outlier, and representation comparison visuals.
- [x] Verify data splits, label maps, save/reload, and controlled component comparisons.

**Gate:** Classification and topic discovery are complete evaluated workflows; neither relies on preassigned labels/groups as its claimed model implementation.

**Implementation evidence:** [L07–L08](./src/content/docs/labs/classification-and-topics.mdx), [CPU execution guide](./examples/applications/discovery/README.md), and [author review](./docs/reviews/e4-discovery.md). Seven real classification methods share a fixed held-out fixture; genuine BERTopic representation/density comparisons and local topic-label generation have CPU records. Optional hosted comparison is documented with an offline importer; no live API execution is claimed. Independent review remains pending.

### E5 — prompting, chains, memory, and agents

**Depends on:** E3 and the generation setup used in E4 where shared.

- [x] Complete Chapter 6–7 work, including chain-of-thought, tree-of-thought, real structured-output constraints, and supported LangChain workflows.
- [x] Complete L09–L11 with verifiable tasks, bounded tools, and controlled prompt comparisons.
- [x] Add explicit state/candidate/memory traces and topic-specific failure exercises.
- [x] Record modernization differences from book-era APIs and mark unsupported optional backends honestly.
- [x] Add acronym aliases and disambiguate chain prompting, chains, reasoning text, and tree search in lessons and graph links.

**Gate:** The learner can compare prompt strategies, inspect a memory policy, and run a terminating tool workflow. Format enforcement and task correctness are measured separately.

**Implementation evidence:** [L09–L11](./src/content/docs/labs/prompting-and-workflows.mdx), [locked local execution guide](./examples/applications/workflows/README.md), and [author review](./docs/reviews/e5-workflows.md). Eighteen detailed lessons connect to real constrained generation, sampled voting, exact bounded search, actual multi-call chains, generated summaries, and terminating tool workflows. Independent teaching acceptance remains pending.

### E6 — retrieval, RAG, and multimodal workflows

**Depends on:** E4–E5; foundation embeddings from E3.

- [x] Complete Chapter 8–9 work, including advanced retrieval techniques and actual multimodal preprocessing.
- [x] Complete L12–L14 and the grounded-document and image/text projects.
- [x] Add a held-out retrieval set, absent-answer cases, citation checks, and stage-specific evaluation.
- [x] Add authentic image input/output records with provenance and separate hardware profiles.
- [x] Upgrade the RAG path and create the multimodal learning route.

**Gate:** Real model workflows run in their supported profiles; sources/citations and processed model inputs are inspectable; recorded-output browsing is clearly labeled.

### E7 — embedding and representation training

**Depends on:** E4/E6 evaluation foundations and the training environment; reuse E2 NER work.

- [x] Complete Chapter 10–11 work, including Augmented SBERT, all required loss explanations, TSDAE, SetFit, continued pretraining, and full NER preparation.
- [x] Complete L15–L20 with baseline comparisons, appropriate data splits, saving/reloading, and held-out evaluation.
- [x] Add training curves, gradient/frozen-parameter checks, alignment visuals, and gold/silver-data provenance.
- [x] Extend the NER reference workflow to the chapter's full teaching scope without duplicating its source implementation.
- [x] Publish measured runtime profiles and recorded results; retain unverified status for any configuration not actually executed.

**Gate:** Each training workflow demonstrably updates the intended parameters, evaluates on untouched data, and produces a reusable artifact. Loss reduction alone is not the completion criterion.

### E8 — generation adaptation and evaluation

**Depends on:** E7's training/evaluation practices; reuse E2 LoRA work and E5 generation setup.

- [ ] Complete Chapter 12 work and L21–L23, including data templates, loss masks, full/PEFT comparison, QLoRA, merging, reward modeling, and DPO.
- [x] Verify the supported float32 CPU and Linux NF4 CPU profiles, with artifact reload/merge checks. GPU/paged optimization remains unverified.
- [x] Add the fixed evaluation suite, human rubric, and optional offline judge-import/calibration workflow. Actual human/hosted ratings remain absent; no calibration result is claimed.
- [x] Implement the instruction/preference adaptation project and generation training path; independent acceptance pending.
- [x] Publish versioned actual CPU run records, model/data references, fresh-kernel notebooks, and honest limits of the trained examples.

**Implementation:** Nine detailed Chapter12 lessons, three staged L21–L23 notebooks, full/LoRA/reward/DPO CPU training, a genuine Linux NF4 profile, five interactive views, fixed evaluation, an offline judgment importer and project/path are delivered. See [author review](./docs/reviews/e8-generation-adaptation.md). The first checklist item remains open for the required supported GPU QLoRA evidence and independent acceptance. All four tiny policies score0/8 generated exact match on the fixed suite; optimization/artifact checks do not establish task success.

**Gate:** Instruction and preference workflows have actual execution evidence and before/after evaluation; unsupported hardware/configurations are not advertised as verified. Explain every step from raw record to loss to saved artifact.

### E9 — integrate discovery, navigation, and learner practice

**Depends on:** E3–E8 content; implement small enabling pieces earlier when required.

- [ ] Complete the glossary, alias-aware search, query regression set, and chapter/content filters where helpful.
- [ ] Add graph relationship reasons, prerequisite routes, lab links, and cross-chapter discovery.
- [ ] Migrate chapter coverage displays to the audited inventory and retire redundant mappings safely.
- [ ] Finish project/path navigation and local progress migration if richer practice tracking was implemented.
- [ ] Review the entire UI across keyboard, touch, small screens, themes, reduced motion, and text alternatives.
- [ ] Measure the production site with the enlarged content/graph and resolve meaningful regressions.

**Gate:** A reader can find a term by acronym, full name, or task, understand its prerequisites, complete its lab route, and return to a related concept without broken navigation.

### E10 — independent review and evidence-backed release

**Depends on:** E0–E9 gates.

- [ ] Reconcile every core/context inventory entry with its destination and review record; resolve all required partial/missing items.
- [ ] Re-run affected examples from clean documented environments; check every published verification badge against its record.
- [ ] Perform technical review of equations, objectives, preprocessing, evaluation, masks, and artifact handling.
- [ ] Pilot representative beginner and intermediate paths, record misconceptions, and revise lessons that fail their objectives.
- [ ] Run the complete site release gate, coverage checks, example manifest checks, and targeted browser acceptance checks.
- [ ] Recheck source attribution, original assets, dataset/model references, external links, and download/notebook entry points.
- [ ] Update README, sources, contribution instructions, release notes, and the public coverage report to match the evidence.

**Gate:** Every required objective has accepted teaching coverage; every required practical workflow has supported execution evidence; no critical correctness or navigation problem remains. Any deferred extension is named separately and does not silently remove a book requirement.

**Release artifact:** Coverage report, review summary, verified-example manifest, known limitations, and a release description grounded in what was actually delivered.

### 10.1 Milestone sizing and sequencing

This is curriculum and software work across 12 chapters. Treat each chapter or cohesive lab as a reviewable batch; do not expand dozens of pages in one unreviewed generation pass.

Use the three reference lessons to measure actual authoring, coding, execution, and review effort. Then estimate remaining batches from observed effort and available hardware. This plan deliberately does not invent a calendar deadline or GPU budget. Infrastructure delay should not prevent independent writing, small mechanism examples, or coverage review, but unavailable execution evidence keeps the relevant gate open.

Priority within each batch:

1. Fix incorrect or misleading existing explanations and claims.
2. Teach missing prerequisites and core mechanisms.
3. Implement and evaluate the book's missing practical workflow.
4. Add the visual and practice that make that mechanism easier to learn.
5. Integrate navigation and search; then consider supplemental extensions.

## 11. Per-change workflow and review packet

For each lesson or lab change:

1. Select source-inventory IDs and state the specific learning gap.
2. Read the relevant book section and official notebook; verify current implementation APIs in primary documentation.
3. Draft objectives, prerequisite explanations, a worked trace, and evaluation before writing polished prose.
4. Implement the smallest honest mechanism example, then the real workflow where required.
5. Execute it; save the environment/configuration, results, errors, and resource measurements.
6. Write the lesson around that evidence and build the matching visual and exercises.
7. Add sources, aliases, graph reasons, and path/project links.
8. Validate only the affected examples plus the required site/coverage checks; broaden testing when shared logic or dependencies change.
9. Obtain editorial/technical review and address findings.
10. Update status and evidence records; only then describe the work as complete.

Each review packet should include:

- What a learner could not do before and can do after.
- Source sections and objectives addressed; any remaining gap.
- Files changed and the source-of-truth example entrypoints.
- Execution environment, command, artifact/run record, and metric interpretation.
- Desktop/mobile views for changed interactive UI and relevant accessibility checks.
- Exercise solutions and the rubric result with reviewer identity/role and date.
- Compatibility notes, skipped configurations, and reasons.

Do not add independent manual approval gates for routine authoring or code changes. Technical review and release evidence are quality requirements; the user can continue implementation batch by batch using this plan.

## 12. Risks and concrete responses

| Risk                                            | Response                                                                                                                    | Completion impact                              |
| ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| A large page count conceals shallow teaching    | Source objectives, worked examples, rubric, and learner tasks replace count-based claims.                                   | Shallow mapped pages remain partial.           |
| Generated prose repeats generic filler          | Require a topic-specific trace, misconception, diagnostic exercise, and implementation.                                     | Editorial gate fails until revised.            |
| Book code uses outdated APIs                    | Preserve the learning objective, verify current docs, pin a working environment, document changes.                          | Unsupported copied code is not a verified lab. |
| GPU memory or runtime is unavailable            | Provide theory, CPU mechanism, and clearly labeled recorded results; schedule a supported full run when hardware exists.    | Practical verification stays open.             |
| Dataset access or licensing changes             | Keep provenance and fetch instructions; use original fixtures and a documented replacement for the same learning objective. | Reverify affected metrics and source mapping.  |
| Small teaching data yields misleading gains     | Label its purpose, use separate realistic evaluation, report failures and sample sizes.                                     | No quality claim from smoke-test metrics.      |
| Train/test leakage invalidates results          | Version split definitions, deduplicate, isolate augmentation, and test split invariants.                                    | Retrain and reevaluate before acceptance.      |
| Model outputs drift across environments         | Pin artifacts, record versions, use invariants/tolerances, preserve run provenance.                                         | Reverify changed configurations.               |
| Graph and lessons disagree                      | Generate from shared canonical metadata and validate destinations/relationships.                                            | Build or coverage gate fails.                  |
| New UI obscures long lessons                    | Progressive depth, stable anchors, text alternatives, and representative reader checks.                                     | Revise interaction/layout before release.      |
| Heavy dependencies make contributions difficult | Separate lab environments and lightweight site checks; publish setup troubleshooting.                                       | Site contribution remains accessible.          |
| Copyright/source confusion                      | Original prose/figures, precise attribution, license records for external code/data, no committed book PDF.                 | Resolve provenance before publication.         |

## 13. Final acceptance checklist

### Coverage and teaching

- [ ] All 12 chapters are audited below the curated-topic level.
- [ ] Every substantive book objective has an accepted mapping, with justified treatment for contextual sections.
- [ ] All existing lessons are reviewed under the new rubric; legacy `full` flags do not bypass review.
- [ ] Every concept has appropriate definitions, mechanism, worked examples, use cases, limitations, practice, and connections.
- [ ] Mathematical symbols, tensor shapes, terminology, and graph relationships are consistent across lessons.
- [ ] Supplemental material is distinguishable from book coverage.

### Implementation and evidence

- [ ] L01–L23 are delivered to their declared scope, including required real-model/training subworkflows.
- [ ] Every displayed executable example is registered and traceable to maintained source.
- [ ] Required notebooks run from a clean kernel in a documented environment.
- [ ] No fabricated model output, benchmark gain, runtime, or verification badge is presented as measured.
- [ ] Baselines, evaluation splits, metrics, failure cases, and reloadable artifacts exist for applied workflows.
- [ ] GPU/API limitations and first-download requirements are visible before the learner runs code.
- [ ] Example verification is separated from ordinary static hosting and does not expose credentials.

### Experience and release

- [ ] Glossary, aliases, chapter coverage, graph, search, and paths agree on canonical destinations.
- [ ] Interactive visuals have meaningful controls, inspectable results, reset, and accessible alternatives.
- [ ] Mobile, keyboard, light/dark, reduced-motion, and performance checks cover representative complex pages.
- [ ] Independent technical review and representative learner trials have produced resolved feedback.
- [ ] README, source attribution, contribution instructions, and public completion claims match the evidence.
- [ ] The release report lists any remaining optional extensions and unsupported configurations clearly.

## 14. Sources and implementation references

The supplied first-edition PDF is the source of the chapter structure and scope. Its detailed contents were checked on PDF pages 7–12; selected body sections were also checked for architectural improvements, BERTopic representations, advanced RAG, and embedding training. The completed E0 initial audit records source locations, embedded objectives, notebook provenance, and review limitations in `docs/audits/`. Book printed page numbers and PDF viewer page numbers remain distinct.

Primary sources checked while preparing this plan:

- [Authors' official book repository and chapter notebooks](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models) — implementation source and comparison target; record the exact revision used when implementing a lab.
- [Transformers chat templates](https://huggingface.co/docs/transformers/chat_templating) — model-specific message formatting and special tokens.
- [Transformers token classification](https://huggingface.co/docs/transformers/tasks/token_classification) — token/label alignment and NER workflow reference.
- [PEFT LoRA documentation](https://huggingface.co/docs/peft/developer_guides/lora) — adapter configuration and supported operations; check the chosen version's reference.
- [TRL supervised fine-tuning](https://huggingface.co/docs/trl/sft_trainer) and [DPO training](https://huggingface.co/docs/trl/dpo_trainer) — current trainer/data contracts to verify during implementation.
- [Sentence Transformers training overview](https://www.sbert.net/docs/sentence_transformer/training_overview.html) — model, dataset, loss, evaluator, and trainer components.
- [BERTopic representation models](https://maartengr.github.io/BERTopic/getting_started/representation/representation.html) — modular topic representations.
- [LangChain overview](https://docs.langchain.com/oss/python/langchain/overview) — current framework entry point; the book's historical calls require compatibility review.
- [Pagefind section weighting](https://pagefind.app/docs/weighting/) — supported search-index weighting approach.

Repository evidence inspected: `plan.md`, `README.md`, `CONTRIBUTING.md`, `src/data/book-map.ts`, `src/data/topic-coverage.json`, `src/content.config.ts`, representative lessons, `src/lib/graph.js`, `tests/content.test.mjs`, `scripts/check-code-examples.mjs`, and `.github/workflows/deploy.yml`.

These references support the implementation direction, not a claim that all proposed configurations are already compatible or tested. Recheck relevant official documentation when selecting the exact environment, and record the verified version beside each runnable example.

## 15. Implementation tracker and immediate next task

| Phase                                 | Status      | Evidence required to close                                                                      |
| ------------------------------------- | ----------- | ----------------------------------------------------------------------------------------------- |
| E0: audit and honest coverage         | Complete    | Source inventory, two generated reports, per-lesson findings, pinned notebooks, and validation. |
| E1: executable-example foundation     | Complete | 100-example registry, two locked environments, runner, synchronized lesson/notebooks, and CPU execution records. |
| E2: three reference lessons | Implemented; review pending | Detailed lessons, interactive labs, CPU model/training records, author review; independent acceptance still open. |
| E3: foundations | Implemented; review pending | 32 rewritten/new lessons plus E2 tokenization, L01–L06 CPU evidence, graph/glossary/path updates; independent acceptance open. |
| E4: classification and topics         | Implemented; review pending | 15 expanded/new lessons, L07–L08 CPU evidence, explorer projects; hosted importer documented, independent acceptance open.                                                    |
| E5: prompting and workflows           | Implemented; review pending | 18 expanded/new lessons; five L09–L11 local-model sources/notebooks, interactive traces, and bounded contracts.                                                    |
| E6: RAG and multimodal                | Implemented; review pending | 23 detailed lessons, five actual-model sources/notebooks, L12–L14 interactives, grounded/image-text projects and routes.                                                    |
| E7: embedding/representation training | Implemented; review pending | 15 detailed lessons; six actual CPU training workflows/notebooks, gradient/freeze/mask checks, held-out task results and reusable artifacts.                                                  |
| E8: generation adaptation | CPU implementation delivered; GPU/review gates open | Nine detailed lessons, L21–L23 actual CPU workflows/notebooks and Linux NF4 evidence; supported GPU QLoRA and independent acceptance remain pending. |
| E9: integrated learner experience     | Not started | Discovery, navigation, practice, and accessibility/performance checks.                          |
| E10: review and release               | Not started | Reconciled coverage, execution evidence, learner feedback, and release report.                  |

**Next:** Collect independent review of E2–E8 and verify supported GPU QLoRA; E9 discovery/navigation integration is the next implementation milestone when authorized. Full enhancement acceptance remains tied to the documented review gates.
