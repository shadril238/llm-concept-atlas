# E3 foundations: implementation and author technical review

Date: 2026-11-20. Reviewer: implementing assistant (author technical pass). **Independent editorial/technical review and beginner pilot are pending.** No independent scores or acceptance identities are invented.

## Scope delivered

Thirty-two foundation lessons were rewritten or added; the E2 tokenization reference is reused. New canonical pages cover model access, sequence-to-sequence, cross-attention, and sequence packing. Existing URLs and reader completion IDs remain stable. The source inventory maps the chapter objectives to canonical pages/subsections and keeps coverage, execution, and independent acceptance separate.

The lessons replace generic process prose with topic-specific definitions, mathematical/shape traces, worked cases, decision boundaries, limitations, and unique self-checks/solutions. Their small Python mechanisms are synchronized from source. Actual model work is shared through L01–L06, with clear labels for random weights, invented numeric values, and recorded pretrained outputs.

## Execution evidence

- L01: real pinned CPU generation, tokenizer/model pairing, prompt/continuation separation, module shapes, and LM-head equality. The small base model repeats itself; this failure is displayed rather than polished away.
- L02: the existing E2 eight-case real-tokenizer package remains canonical. Numerical/whitespace input counts additionally appear in L03.
- L03: actual pretrained GloVe neighbors with artifact checksum; same-row/different-context checks for bank; mask-aware sentence pooling/cosines; eight-word PCA with retained-variance and original-distance limitations.
- L04: learned skip-gram song vectors, original split sessions with unique song sets, train-only popularity baseline, exclusion of observed songs, held-out Recall@3/MRR, cold-start boundary, and saved-vector reload. The optional author-published Last.fm route is documented and its parser tested; no full-data model result is claimed.
- L05: explicit random-weight lookup, pre-normalization, Q/K/V projection, causal attention, head/output projection, residual/MLP, and LM logits. Attention agrees with PyTorch's reference; future-value perturbations cannot affect earlier causal outputs.
- L06: stable softmax, positive temperature, top-k then top-p with crossing-token retention, seeded sampling, real cached/uncached logit agreement, exact cache tensor bytes, warm-up, and three same-token CPU timing runs. The strict check and timing use float64 and are not a production-precision/GPU throughput claim.

Initial generation inspection exposed an inference-tensor/autograd boundary error; the projection was moved into inference mode. Initial float32 cache comparisons showed amplified rounding differences; the retained failure led to a declared float64 reference with a strict 1e-9 bound. Successful evidence comes from rerunning the corrected source, not changing failures to passes.

## Curriculum and UI integration

The sentence-processing path now names actual fixed tokenizer cases and produces concrete artifacts/checkpoints. Graph and directory search use aliases; a glossary links canonical terms; prerequisite reasons explain dependencies; chapter source tables show objective destinations without implying acceptance. The graph inspector links the connected labs rather than adding an overwhelming lab-node layer.

Interactive experiments expose attention masks/rows, actual representation snapshots, RoPE pairs, decoding filters, cache estimates, architecture contracts, and recorded recommendation rankings. Controls state whether they calculate or select recorded data. Numerical tests compare browser and Python quantities, including boundary masks and relative-position invariants.

## Author review findings and limits

Browser checks verified calculated packed-attention rows, top-k probability renormalization, cache memory changes with KV-head count, relative-position invariance in RoPE, actual representation snapshots, and GQA alias search with prerequisite reasons and a lab link. The RoPE experiment was inspected at a 390-pixel mobile viewport in both light and dark themes without horizontal page overflow; desktop experiment layouts were also inspected. These are author UI checks, not a beginner usability study.

- Definitions and prerequisites distinguish IDs, lookup rows, contextual states, sentence vectors, logits, probabilities, and task outputs.
- Random-weight and small-model runs validate different claims. Neither is described as broad task competence.
- Scope follows the book's Chapters 1–3 and pinned official notebooks. Model sizes, original fixtures, PCA view, and CPU precision choices are atlas adaptations.
- Synthetic playlist themes make the ranking task easy; results do not establish real-user preference quality or fairness.
- CPU/cache/storage estimates identify what is measured versus theoretical; cache bytes exclude weights/activations/allocator overhead.
- Independent readers must still verify that prerequisite bridges, shapes, exercises, and interactive controls are understandable without unstated knowledge.
- A larger model, dataset, dtype, device, or kernel needs its own execution evidence; existing records cannot be transferred by name alone.

See [E3 execution guide](../../examples/training/reference/E3.md), [verification report](../../examples/verification/README.md), and [reference template](../templates/reference-lesson.md).
