# E4 classification and topic discovery: author technical pass

Date: 2026-11-20. Reviewer: implementing assistant (author review). **Independent technical/editorial review and beginner pilot remain pending.** Execution, source mapping, and teaching acceptance are separate; no independent scores are invented.

## Delivered scope

Fifteen specific lessons were rewritten or added across Chapters 4–5, including new canonical c-TF-IDF and MMR pages. Existing URLs and completion IDs remain stable. The classification project is now an evaluated workflow rather than an unpinned inference sketch; the new research collection explorer links document evidence, topic descriptions, outliers, similarities, and a controlled density comparison. L07/L08 have shared source/notebook/lesson views and pinned public model/data provenance.

## Source review and adaptations

Complete Chapter 4–5 extracted text and pinned official notebook stages were inspected. Representative PDF pages 141, 151, 172, and 181 (printed 119, 129, 150, 159) were visually inspected. The book's Rotten Tomatoes/arXiv workflows inform scope; UCI movie sentences and original fictional research summaries are declared atlas adaptations. No source PDF or copied artwork is published.

The lesson distinguishes macro-F1 from support-weighted F1. The printed discussion of equal treatment must not be generalized from balanced class counts to unequal support. It also states T5 depth as checkpoint-dependent, uses strict invalid-output parsing rather than mapping everything except negative to positive, and distinguishes cosine similarity from an angle or calibrated correctness probability.

## Execution and behavioral evidence

- Seven actual classification methods share 80 held-out UCI IDs and 12 separately reported original challenge IDs. Train-only TF-IDF/head/prototypes and validation-only C/threshold selection are explicit. Label maps and NLI entailment columns are inspected rather than guessed.
- Frozen encoder pooling uses masks, L2 normalization, finite/unit-norm checks, and a fixed input cap. CPU method timings use batch 16, warm-up, three measured passes, and thread limits; model loading is excluded.
- Generative labels are parsed exactly after trim/case-fold. Invalids remain in recall/accuracy denominators. Their absence in this one recorded test run is not a format guarantee.
- The saved JSON embedding head is reloaded and its sigmoid probabilities checked against the fitted classifier. The inference command uses the exact encoder contract and emits a review flag.
- BERTopic actually fits embeddings → UMAP → HDBSCAN → class counts → c-TF-IDF. It does not receive preassigned theme groups. Three topics and one noise point occur in the recorded baseline.
- Actual KeyBERTInspired and MMR updates assert fixed assignments. A separate min_samples-only comparison holds embeddings and reduction fixed, reports partition transitions and noise, and uses label-invariant ARI under an explicit noise convention.
- The c-TF-IDF hand formula is checked against the real library; tests also cover unequal class lengths and the integer mean convention. Metrics, MMR endpoints, invalid outputs, threshold-empty cases, and split uniqueness have behavioral tests. Browser/Python calculations are compared within floating-point tolerance.
- Safetensors reload verifies stored topic data and c-TF-IDF. Its lightweight cosine inference is checked and explicitly distinguished from the original fitted density estimator; agreement is reported rather than assumed.
- Local TextGeneration uses actual FLAN-T5 and records prompts, unedited outputs, fixed assignments, and the parent artifact hash. Author inspection rejects the noise name and narrow translation name, and notes majority-label limitations.
- Optional hosted comparison is a documented workflow plus offline importer, not a live API client. Synthetic importer tests cover missing outputs, provider errors, bad/duplicate IDs, and invalid JSON labels. No paid API result or cost measurement is claimed.

The first local NumPy 2.2.6 run emitted platform matrix-operation warnings. E4 was isolated and rerun under NumPy 2.3.5 with finite, probability, library, and independent tensor/einsum checks retained. The final records come from rerunning final source/lock, not from editing outcomes. BERTopic's custom-backend save and explicit-update warnings correspond to declared choices; the backend is explicitly supplied on reload and no later topic-reduction step is performed.

## Remaining review and limitations

The UCI subset is small and balanced. Source-review/reviewer IDs and pretrained overlap are unknown. Challenge labels can be ambiguous. Different pretrained methods have different prior training, so the method table is not an architecture-controlled leaderboard. The fictional corpus makes themes easier and retains unrelated members to expose limits; it is not an evaluation of scientific-literature trends.

A beginner reviewer should inspect whether the inference contracts, metric denominators, c-TF-IDF factors, separate projection spaces, representation-only invariants, and reload distinction are understandable without unstated background. Independent experts should inspect source-to-lesson coverage, data/annotation choices, generated-label support, and the operational review policy. Larger datasets, other checkpoints, new devices, and live hosted clients require their own evidence.

See [execution guide](../../examples/applications/discovery/README.md), [book gap report](../audits/book-gap-report.md), and [example records](../../examples/verification/README.md).

## Browser verification

Author browser checks verified c-TF-IDF counts/factors, exact invalid-label parsing, real confusion-cell errors/disagreements, the zero-accepted threshold state, outlier selection and the empty changed-density filter state, KeyBERT terms/generated labels, the density transition/ARI/reload report, MMR keyboard endpoints, and MMR alias search with a connected-lab link in the graph. At a 390-pixel mobile viewport, light/dark layouts remained readable without horizontal page overflow; wide metric tables scroll within their containers. A filter/evidence mismatch and uneven metric-card spacing found during inspection were fixed and rechecked. Theme Auto and the normal browser viewport were restored. These checks are not an independent beginner usability study.
