# Reference lesson authoring template

Use the tokenization, LoRA, and named-entity-recognition pages as concrete examples. Keep stable concept IDs and URLs. Do not fill headings with generic prose to satisfy a length check.

1. **Problem and definition:** name a situation, the concept's role, and a nearby concept it is commonly confused with.
2. **Objectives and prerequisites:** list observable abilities; explain the smallest missing mathematical/data bridge and add a recall prompt.
3. **Mechanism:** define symbols, dimensions, units, frozen versus changing state, and the transformation at each step.
4. **Worked trace:** use one complete input with intermediate values; keep the browser and source consistent.
5. **Experiment:** ask for a prediction, allow a meaningful intervention, expose intermediate state, label calculated/invented/recorded data, provide reset and a text equivalent.
6. **Implementation:** register source regions, pinned models, locked environment, data provenance/splits, contracts, timeout, and validation. Give the command before the code and identify excerpts that need their surrounding module.
7. **Evaluation:** compare an appropriate baseline on the same held-out cases, show denominators and errors, and explain what a metric does not establish. Validate artifacts and save/reload.
8. **Decisions and failures:** describe when to choose the method, alternatives, and a specific failure with a detection procedure.
9. **Practice:** include recall, calculation/prediction, debugging, and transfer prompts with hints and reasoned solutions.
10. **Sources and evidence:** identify exact book locations and implementation extensions; link primary sources, source code, and execution records. Keep author review separate from independent acceptance.

Use `ReferenceBrief` for the opening contract and `VerifiedRun` for actual execution metadata. Add `{/* example:ID:start */}` / `{/* example:ID:end */}` markers for synchronized source, then run `npm run examples:sync`. Never invent outputs, resource measurements, review identities, or rubric scores. New local helper imports belong in `supportFiles`; published result data must match hashed artifacts. Reassess audit findings before refreshing lesson hashes.
