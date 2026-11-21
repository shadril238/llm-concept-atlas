"""L09: predeclared prompt comparisons and genuine native schema constraints."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas_workflows.runtime import LocalModel, fixture, messages, save
from atlas_foundations.e5.contracts import ORDER_SCHEMA, extract_answer, validate_order

VARIANTS = [
    "direct",
    "step-instruction",
    "worked-example",
    "few-shot-balanced",
    "few-shot-reversed",
    "few-shot-add-only",
]


def arithmetic_messages(case, variant, design):
    step = variant in ["step-instruction", "worked-example"]
    instruction = (
        "Give one short checkable calculation. End with a separate line Answer: INTEGER."
        if step
        else "Return only one integer, with no explanation or other text."
    )
    result = [{"role": "system", "content": instruction}]
    ids = []
    if variant == "worked-example":
        examples = design[:1]
    elif variant.startswith("few-shot"):
        examples = (
            [design[0], design[2]] if variant == "few-shot-add-only" else design[:2]
        )
        if variant == "few-shot-reversed":
            examples = list(reversed(examples))
    else:
        examples = []
    for example in examples:
        result += [
            {"role": "user", "content": example["question"]},
            {
                "role": "assistant",
                "content": f"{example['worked']}\nAnswer: {example['answer']}"
                if step
                else str(example["answer"]),
            },
        ]
        ids.append(example["id"])
    result.append({"role": "user", "content": case["question"]})
    return result, ids, "answer-line" if step else "integer"


def main():
    model = LocalModel()
    data = fixture()
    design = [r for r in data["arithmetic"] if r["split"] == "design"]
    evaluation = [r for r in data["arithmetic"] if r["split"] == "evaluation"]
    arithmetic = []
    for variant in VARIANTS:
        for row in evaluation:
            assembled, example_ids, mode = arithmetic_messages(row, variant, design)
            call = model.generate(assembled, max_tokens=64)
            strict = extract_answer(call["raw"], mode)
            normalized = extract_answer(call["raw"], "integer")
            if normalized is None:
                normalized = extract_answer(call["raw"], "answer-line")
            arithmetic.append(
                {
                    "id": row["id"],
                    "variant": variant,
                    "question": row["question"],
                    "expected": row["answer"],
                    "reference_calculation": row["worked"],
                    "example_ids": example_ids,
                    "format_contract": mode,
                    "format_valid": strict is not None,
                    "normalized_answer": normalized,
                    "answer_correct": normalized == row["answer"],
                    "call": call,
                }
            )
    extraction = []
    schema_design = next(r for r in data["extraction"] if r["split"] == "design")
    instruction = "Extract the current order. Return exactly one JSON object with keys item, quantity, priority. item is book or pen; quantity is an integer 1 through 9; priority is normal or urgent. Standard delivery and no urgency mean normal. No extra fields or prose."
    for variant in ["instruction-only", "format-example", "native-schema"]:
        for row in data["extraction"]:
            if row["split"] != "evaluation":
                continue
            assembled = [{"role": "system", "content": instruction}]
            if variant == "format-example":
                assembled += [
                    {"role": "user", "content": schema_design["text"]},
                    {
                        "role": "assistant",
                        "content": json.dumps(schema_design["expected"]),
                    },
                ]
            assembled.append({"role": "user", "content": row["text"]})
            call = model.generate(
                assembled,
                max_tokens=128,
                schema=ORDER_SCHEMA if variant == "native-schema" else None,
            )
            checked = validate_order(call["raw"])
            extraction.append(
                {
                    "id": row["id"],
                    "variant": variant,
                    "text": row["text"],
                    "expected": row["expected"],
                    "example_ids": [schema_design["id"]]
                    if variant == "format-example"
                    else [],
                    "validation": checked,
                    "answer_correct": checked["valid"]
                    and checked["value"] == row["expected"],
                    "call": call,
                }
            )
    summary = []
    for group, rows in [("arithmetic", arithmetic), ("extraction", extraction)]:
        for variant in dict.fromkeys(row["variant"] for row in rows):
            selected = [row for row in rows if row["variant"] == variant]
            summary.append(
                {
                    "suite": group,
                    "variant": variant,
                    "cases": len(selected),
                    "answer_accuracy": sum(row["answer_correct"] for row in selected)
                    / len(selected),
                    "format_validity": sum(
                        row["format_valid"]
                        if group == "arithmetic"
                        else row["validation"]["valid"]
                        for row in selected
                    )
                    / len(selected),
                    "prompt_tokens": sum(
                        row["call"]["prompt_tokens"] for row in selected
                    ),
                    "output_tokens": sum(
                        row["call"]["output_tokens"] for row in selected
                    ),
                    "seconds": sum(row["call"]["seconds"] for row in selected),
                }
            )
    assert all(
        set(row["example_ids"]) <= {d["id"] for d in design} for row in arithmetic
    )
    assert all(
        len({r["id"] for r in arithmetic if r["variant"] == v}) == 8 for v in VARIANTS
    )
    # Validity is reported, not silently repaired or treated as correctness.
    native_valid = sum(
        row["validation"]["valid"]
        for row in extraction
        if row["variant"] == "native-schema"
    )
    assert native_valid == 9, (
        "Native decoding failed to complete the supported schema within its output budget"
    )
    for row in extraction:
        if row["variant"] == "native-schema":
            baseline = next(
                r
                for r in extraction
                if r["id"] == row["id"] and r["variant"] == "instruction-only"
            )
            assert baseline["call"]["messages"] == row["call"]["messages"]
    truncation = model.generate(
        extraction[-1]["call"]["messages"], max_tokens=1, schema=ORDER_SCHEMA
    )
    assert (
        not validate_order(truncation["raw"])["valid"]
        and truncation["finish_reason"] == "length"
    )
    result = {
        "model": model.metadata,
        "fixture": data["id"],
        "arithmetic": arithmetic,
        "extraction": extraction,
        "summary": summary,
        "schema": ORDER_SCHEMA,
        "truncation_probe": truncation,
        "design_evaluation_disjoint": True,
        "native_constraint_verified": True,
        "measurement": "One sequential run per prompt/case; native token usage and wall seconds. Reused model/prefix cache; no financial cost or production speed ranking.",
        "grading": "Answer extraction accepts a complete integer or a terminal Answer: integer line. Requested-format validity is checked separately. Other prose is not heuristically repaired. Structured correctness requires a valid exact-key object equal to the reference.",
        "limitations": [
            "Eight arithmetic and nine extraction evaluation cases; no general reasoning benchmark.",
            "A small Q8 model can fail simple arithmetic and produce plausible explanations.",
            "Native grammar constrains structure, not factual correctness; truncated output can still be incomplete.",
            "Examples are predeclared design cases; no variant is selected using final evaluation results.",
        ],
    }
    save("e5-prompts", result)
    print(
        json.dumps(
            {
                "evaluation_cases": 17,
                "arithmetic_variants": len(VARIANTS),
                "native_constraint_verified": True,
                "design_evaluation_disjoint": True,
                "summary": summary,
            }
        )
    )


if __name__ == "__main__":
    main()
