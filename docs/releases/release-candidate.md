# Enhancement release candidate — review pending

Prepared 2026-11-25. This is a draft handoff for the GitHub-hosted atlas, not a published reviewed release. Baseline before E10 preparation: `8484e9690706aa0e55d5fc2be6b031542639ba65` (E9). The reviewed release commit and CI/deployment URLs must be filled after final review; this file does not self-certify its own commit.

## Delivered

The atlas connects 113 original concept lessons with definitions, theory, worked examples, code, use cases, exercises and interactive experiments. Search supports aliases, task phrases and separate full-text results. The graph explains parent, prerequisite and related-topic connections; eight study routes connect reading to labs and practice. Browser-local progress separates reading from self-reported practice.

The source inventory maps all 160 counted book objectives. There are 239 registered examples, 236 with passing records; many are deliberately small calculations, while model workflows retain their own locked environments and execution scope. The build produces 156 pages. See the [verification manifest](../../examples/verification/README.md), [gap report](../audits/book-gap-report.md), [machine audit](../reviews/e10-evidence-audit.json) and [E10 packet](../reviews/e10-review-packet.md).

E10 preparation reconciles eight stale foundation execution statuses against existing authentic records, taking recorded practical objectives from 41/51 to 49/51. It adds explicit scope limits and removes duplicate API-RAG mappings. No model run is redated or inferred from a UI test. A reproducible evidence report is now checked by the release build.

## Validation recorded during preparation

The automated release gate passes: 118 behavioral tests (64 Python, 54 Node), 202 offline examples, coverage/source and execution-record checks, Astro analysis/build, 17,276 internal links/assets, 20 discovery query cases, all 113 concept practice destinations, eight learning routes and production payload budgets. These automation checks validate their specific contracts, not independent comprehension or GPU performance. E9 author browser observations are retained in the [dated E9 review](../reviews/e9-learning-experience.md); a final deployed E10 acceptance check remains a separate step.

## Known limits and release gates

- Independent acceptance is recorded for 0/160 objectives. Actual reviewer and learner evidence must be attached before claiming full enhancement completion.
- Optional hosted classification and hosted API grounding are documented workflows/offline importers. No paid provider execution is claimed; both practical objectives remain unverified.
- Linux CPU NF4 is verified in its declared profile. Supported GPU QLoRA, paged CUDA optimization and book-scale model/data quality remain unverified.
- Three legacy scripts have no passing registry records. Their separately verified successors do not certify those scripts. The twelve official book notebooks were statically inspected, not executed.
- Small models and original fixtures are teaching configurations. Optimization, representation or numerical correctness does not establish application quality, fairness or generalization. Recorded failures remain visible.
- Final independent accessibility/learner trials and dated external-source/license/download checks remain open. Automated internal links do not verify every external service.
- The book's text, PDF and artwork are not redistributed. Original site code is MIT; linked models, datasets and third-party work retain their own licenses and attribution.

## Evidence to attach before a reviewed release

- Reviewed commit/tag and exact scope (full acceptance versus limited preview).
- Independent technical/editorial records with objective IDs, rubric findings and resolved corrections.
- Beginner/intermediate learner observations and revision/retest results.
- Supported GPU run records if claiming the original full enhancement gate.
- Final clean-run, CI, deployment and browser-check links; dated attribution/external-reference findings.
- Updated public coverage and known limitations matching those records.

No final version/tag or independent approval is invented here. Use the [review packet](../reviews/e10-review-packet.md) to close the remaining gates.
