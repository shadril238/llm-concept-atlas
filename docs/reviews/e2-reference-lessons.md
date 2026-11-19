# E2 implementation and author technical review

Date: 2026-11-20. Reviewer: implementing assistant, author technical pass. **Independent technical/editorial review and beginner pilot are pending.** This document does not award enhancement acceptance or claim independent rubric scores.

## Delivered reference pattern

Three original lessons keep their stable IDs, URLs, graph relationships, and reader completion storage. Tokenization compares actual pinned vocabularies; LoRA connects the shaped equation to arithmetic, optimization, and a pretrained adapter; NER connects annotation to actual subword alignment, training, and sentence-preserving span metrics.

Each has motivation, observable objectives, prerequisites, definitions, mathematical/shape traces, recorded or calculated experiments, supported commands, actual results, failure analysis, and recall/calculation/debugging/transfer exercises with solutions. `ReferenceBrief`, `VerifiedRun`, and `ReferenceResults` supply the shared layout the lessons need. Source regions generate fenced code; four notebooks import the implementations. The initial concept examples remain downloadable historical sources and retain their original output checks, but the three lesson pages now point to their replacements.

## Evidence by teaching criterion

| Criterion | Author evidence | Remaining independent check |
| --- | --- | --- |
| Definition and motivation | Concrete context overflow, adaptation, and structured-extraction problems | Ask a beginner to state each problem in their own words. |
| Prerequisite accessibility | Recall prompts and explanations of IDs, shapes, rank, gradients, and word/subword differences | Check whether matrix and gradient bridges are sufficient. |
| Mechanism and mathematics | Explicit LoRA orientation/scaling/counts; token-label masks; exact typed span formulas | Independently check equations and decoder policy. |
| Worked examples | Real tokenizer records; shared LoRA numeric trace; manually scored NER boundary/type cases | Recalculate without relying on the implementation. |
| Implementation and reproducibility | Locked CPU environment, pinned revisions, original split fixtures, run records, checkpoint reload | Reproduce on another supported host; broaden realistic data later. |
| Visual clarity | Piece/ID/offset inspector, calculated factorization, alignment table, span-error experiment | Verify desktop/mobile keyboard use and whether labels explain interventions. |
| Use cases and limitations | Alternatives, memory/data tradeoffs, fixture scope, quantization/domain limits | Check that readers do not infer production quality from a smoke run. |
| Practice and feedback | Topic-specific recall, prediction, debugging, and transfer with worked solutions | Observe actual learner attempts and revise confusing questions. |
| Source traceability | Book printed sections, pinned official notebooks, primary API/model/paper references | Reconcile remaining chapter-level objectives in E3/E7/E8. |
| Accessibility and navigation | Native controls, visible labels, text alternatives, stable routes and source links | Independent keyboard/screen-reader evaluation remains useful. |

## Execution contracts and observations

- `tokenizer-comparison`: two tokenizers, eight recorded inputs, actual IDs/offsets, masks, empty input, and lossless GPT-2 fixture round trips. BERT normalization is deliberately not assigned a lossless contract.
- `lora-trace`: the plan's exact 2-by-2 trace yields Ax = 0.5, update [0.05, 0.10], output [2.05, 1.10]. The browser uses the same quantities.
- `lora-tiny-training`: frozen/full/adapter comparison, first-step A/B gradients, held-out vectors, unchanged base, merge and reload assertions.
- `lora-pretrained`: real Pythia-14m/PEFT optimization, answer-only loss masks, validation-selected checkpoint, fixed test prompts, frozen-parameter digest, two adapter reloads, float32/float64 merge diagnostics, and observed greedy output.
- `ner-span-scoring`: exact word spans preserve sentence IDs and are checked against hand-scored boundary/type cases and the browser implementation.
- `ner-training`: actual tiny-BERT optimization; separate tokenizer revision and vocabulary check; first-subtoken alignment; specials/padding ignored; truncated records rejected; embeddings frozen; validation selection; all-O baseline and real test predictions; checkpoint reload; actual validation errors published.

The recorded NER test fixture is intentionally easy: shared sentence templates and disjoint name sets. Its perfect test result is displayed with denominators and limitations, alongside validation errors including a false positive on an entity-free sentence. LoRA's answer loss and observed completion belong to a tiny authored completion task. Neither result is a realistic general quality benchmark.

The first float32 merge assertion was too strict near zero logits. The failed record is retained; the corrected check records scale-relative float32 drift and independently verifies the merge algebra in float64 below 1e-9 maximum absolute error. The local module originally named `tokenizers.py` shadowed the external package; it was renamed before successful runs. These failures were repaired and rerun, not relabeled as passes.

## Author browser verification

Checked the built lessons at desktop width 1265 and mobile width 390. Tokenizer/model/input selection and token inspection update recorded IDs/counts; keyboard changes to the LoRA slider and rank selector recalculate outputs; NER supervision policy and prediction cases update labels and strict-span F1. Reset controls work. The mobile NER document has no page-wide horizontal overflow; the alignment table scrolls within its container. An inherited sibling-margin issue was corrected and the final experiment layout rechecked.

## Review status and next work

E2 implementation and author technical checks are delivered. All three lessons remain **author-reviewed / awaiting independent acceptance**. No coverage entry is promoted to independent enhancement acceptance by this change. E3 can use the reference pattern for implementation, while independent feedback and the beginner pilot remain open quality work. E8 extends the initial adapter work to broader instruction data, QLoRA, full fine-tuning comparisons, merging configurations, and evaluations.

See [execution records](../../examples/verification/README.md), [reference environment](../../examples/training/reference/README.md), and [lesson template](../templates/reference-lesson.md).
