# E5 prompting and application workflows: author technical pass

Date: 2026-11-21. Reviewer: implementing assistant, author review. Independent technical/editorial review and beginner pilot remain pending. Execution, source coverage, and teaching acceptance are separate; no independent scores or acceptance claims are invented.

## Scope and source review

Eighteen Chapter 6–7 lessons are expanded, including new canonical chain-of-thought and tree-of-thought pages. Each has specific definition, theory, worked example, interactive evidence, implementation, applications, limitations, and two topic-specific self-checks. Existing URLs/completion IDs are retained. L09–L11 include five runnable sources/notebook views in one isolated pinned environment.

Complete Chapter 6–7 extracted text and pinned official notebook stages were inspected. Representative PDF pages 208, 212, 217, 224, 235, 237, 241, and 245 were visually inspected. Original tasks and visuals replace book examples/artwork. The source PDF is not redistributed. Modernization notes correct the distinction between GGUF container and tensor precision: a book-era FP16 filename is not evidence for eight-bit weights. The actual Q8 artifact, embedded chat template, and metadata are checked.

## Behavioral and execution evidence

Actual native generation compares predeclared arithmetic/extraction prompts with design/evaluation separation. Strict extracted-answer metrics are distinguished from human interpretation of prose. Native JSON-schema decoding is exercised with identical messages to the instruction baseline and a deliberate truncation failure. Format correctness is separate from content correctness; 9/9 valid order objects include only 2/9 fully correct objects.

Repeated local samples preserve raw text, invalid extraction, unique plurality, ties/abstention, and a separate all-sample strict-majority flag. The two sampling paths change prompt and decoding and are not presented as a single-variable comparison. Exact rational search includes legal-state generation, duplicate handling, global beam selection, node budgets, stop reasons, AST/leaf verification, and full local traces. A restricted actual model ranker chooses among at most eight legal candidates for at most three calls. A diagram labels its displayed subset rather than pretending every search node is drawn.

Actual LangChain core Runnable/typed-message adapters match plain-Python serialized inputs and greedy outputs. A real two-call extraction/pricing/copy chain preserves its failed end-to-end result. Memory comparisons use the same conversation, actual tokenizer counts, explicit reserve, whole-pair evictions, actual summary updates, source IDs, and separate retention/use diagnostics. At the larger budget the recorded buffer/window/summary field scores are 3/1/0 out of three; this is not a universal policy ranking.

Real finite-choice tool decisions execute plain-Python and actual StructuredTool functions with explicit observations. Strict argument validation, invalid recovery, observation length, required tool-before-final, and max-step stopping are tested. Scripted/FakeListLLM paths are labeled test doubles; real model paths preserve incorrect tool/final choices. The restriction to predeclared operands, finite action IDs, and a common final-value domain is explicit. It is not an unrestricted autonomous agent or arbitrary operand-extraction demonstration.

Offline contracts test strict JSON ambiguity/types, parser boundaries, invalid vote denominators, ties, exact fractional solving, forbidden expressions, pruning/budget distinction, finite tool operands, allowlists, bounded termination, and memory packing. Browser/Python validators and votes are checked against each other. All five notebooks execute all four code cells in fresh actual Jupyter kernels via pinned nbclient/ipykernel, with setup, fixture inspection, predictions, failure analysis, artifact reload, and transfer exercises. This verifies Restart → Run All execution, not independent notebook-editor usability. Final model records are created after source formatting and notebook execution so published artifact hashes bind actual final-source runs.

## Native runtime and limits

The locked llama-cpp-python 0.3.37 macOS build initializes Metal even with zero GPU layers. The managed sandbox rejected queue initialization; the identical host run succeeded. Native diagnostics confirmed CPU KV and compute buffers and zero Metal compute allocation with explicit CPU settings. No source patch or manual record editing was used. CPU computation is not equated with a Metal-free binary. The tiny model is weak even on elementary tasks, and compilation/download time is separate from generation timing.

The original fixture is small, synthetic, and author written. No broad reasoning ranking, precision-loss comparison, peak-memory reduction, independent speed claim, hosted API result, or production agent capability is established. Independent reviewers should inspect mathematical/search contracts, framework modernization, metric denominators, source mapping, and whether the worked explanations help beginners. Beginner feedback should test the distinctions between explanations/calls/search, validity/truth, stored/used facts, and fake/real tool paths.

## Browser and release verification

Author browser checks exercised actual complete/beam/node-budget search results, exact fractional solutions, recorded native-format versus content scores, wrong-but-valid and duplicate-key JSON, invalid/tied voting, buffer/window/generated-summary fields and evictions, real wrong tool choices, fake invalid-argument recovery, and max-step termination. The reset control was rechecked through keyboard activation. CoT, ToT, and LCEL aliases locate concepts in the graph; the CoT inspector shows specific prerequisite reasons and the connected-lab route. The final lab reported no browser console errors.

At a 390-pixel viewport, memory controls and metric cards remained readable with no horizontal page overflow in light and dark themes. The normal viewport and Auto theme were restored. Uneven metric-card margins found during inspection were corrected, and the long child-branch list was collapsed for readability. These are author checks, not an independent beginner study.

The final release gate passed 67 tests (29 JavaScript / 38 Python), 163 offline foundation examples, source/registry synchronization, coverage checks, Astro diagnostics, the 135-page static build, and 11,414 internal links/assets. Five actual model workflows have final-source/lock execution records, and all five notebooks passed all four code cells in fresh actual kernels. The registry contains 185 examples; the atlas has 104 concept lessons. Independent teaching acceptance remains pending.

 See the [execution guide](../../examples/applications/workflows/README.md) and [run records](../../examples/verification/README.md).
