"""L10: real sampled answers, exact bounded search, and model-guided candidate ranking."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas_workflows.runtime import ROOT, LocalModel, fixture, messages, save
from atlas_workflows.prompts import arithmetic_messages
from atlas_foundations.e5.contracts import extract_answer, vote
from atlas_foundations.e5.search import search


def main():
    model = LocalModel()
    data = fixture()
    sampled = []
    design = [r for r in data["arithmetic"] if r["split"] == "design"]
    for case in [r for r in data["arithmetic"] if r["split"] == "evaluation"][:4]:
        assembled, _, _ = arithmetic_messages(case, "step-instruction", design)
        samples = []
        for seed in [101, 102, 103, 104, 105]:
            call = model.generate(assembled, max_tokens=64, temperature=0.7, seed=seed)
            answer = extract_answer(call["raw"], "integer")
            if answer is None:
                answer = extract_answer(call["raw"], "answer-line")
            samples.append(
                {"answer": answer, "correct": answer == case["answer"], "call": call}
            )
        aggregation = vote([row["answer"] for row in samples])
        sampled.append(
            {
                "id": case["id"],
                "variant": "format-probe",
                "question": case["question"],
                "expected": case["answer"],
                "samples": samples,
                "aggregation": aggregation,
                "aggregate_correct": aggregation["winner"] == case["answer"],
            }
        )
        constrained = []
        for seed in [101, 102, 103, 104, 105]:
            call = model.generate(
                messages(
                    case["question"],
                    "Solve the arithmetic task. Return only one integer.",
                ),
                max_tokens=8,
                temperature=0.7,
                seed=seed,
                schema={"type": "integer", "enum": list(range(101))},
            )
            answer = json.loads(call["raw"])
            constrained.append(
                {"answer": answer, "correct": answer == case["answer"], "call": call}
            )
        aggregation = vote([row["answer"] for row in constrained])
        sampled.append(
            {
                "id": case["id"],
                "variant": "bounded-integer",
                "question": case["question"],
                "expected": case["answer"],
                "samples": constrained,
                "aggregation": aggregation,
                "aggregate_correct": aggregation["winner"] == case["answer"],
            }
        )
    games = []
    trace_dir = ROOT / "examples/artifacts/e5-search"
    trace_dir.mkdir(parents=True, exist_ok=True)
    for case in data["games"]:
        for strategy, beam, budget in [
            ("complete", None, 4000),
            ("beam-1", 1, 4000),
            ("beam-4", 4, 4000),
            ("budget-16", None, 16),
        ]:
            outcome = search(case["numbers"], beam=beam, max_nodes=budget)
            full = outcome.pop("full_trace")
            name = f"{case['id']}-{strategy}.json"
            (trace_dir / name).write_text(json.dumps(full, indent=2) + "\n")
            games.append(
                dict(id=case["id"], strategy=strategy, full_trace_file=name, **outcome)
            )
    model_choices = []

    def ranker(candidates, state):
        options = "\n".join(
            f"{i}: {action}; remaining {', '.join(str(v) for v, _ in next_state)}"
            for i, (next_state, action) in enumerate(candidates)
        )
        call = model.generate(
            messages(
                f"Game 24: combine all four input numbers exactly once using + - * /. Current remaining values: {state['values']}. Choose a next candidate that may lead to 24. Return only its integer index.\n{options}"
            ),
            max_tokens=8,
            schema={"type": "integer", "enum": list(range(len(candidates)))},
        )
        selected = json.loads(call["raw"])
        model_choices.append(
            {
                "state": state["values"],
                "options": [action for _, action in candidates],
                "selected": selected,
                "call": call,
            }
        )
        return selected

    guided = search(data["games"][0]["numbers"], beam=1, ranker=ranker)
    (trace_dir / "model-guided.json").write_text(
        json.dumps(guided.pop("full_trace"), indent=2) + "\n"
    )
    assert len(model_choices) <= 3
    assert all(not row["solution"] or row["verified"] for row in games)
    result = {
        "model": model.metadata,
        "samples": sampled,
        "vote_policy": "Unique highest valid count; ties/empty votes abstain. strict_majority separately counts all samples, including invalids. Correlated samples can agree on a wrong answer.",
        "sampling_paths": "Two mechanism probes, not an isolated intervention comparison: a free-format calculation prompt and a direct integer prompt with native enum grammar over the same predeclared 0..100 domain. Neither uses a gold-only answer domain.",
        "games": games,
        "guided": guided,
        "model_choices": model_choices,
        "search_verified": True,
        "model_path_executed": True,
        "limits": {
            "full_node_budget": 4000,
            "display_first_nodes": 100,
            "model_rank_pool": 8,
            "model_max_calls": 3,
        },
        "limitations": [
            "Exact arithmetic solver, not a language-model reasoning benchmark.",
            "Model variant ranks a pre-generated legal candidate pool; it does not invent or validate arithmetic operators.",
            "Beam pruning can discard every solution. Node-budget exhaustion is not proof of impossibility.",
            "The displayed trace is a declared subset plus solution ancestors; complete trace files remain local.",
            "Visible model text is an output artifact, not access to hidden computation.",
        ],
    }
    save("e5-search", result)
    print(
        json.dumps(
            {
                "search_verified": True,
                "model_path_executed": True,
                "sampled_cases": len(sampled),
                "model_calls": len(model_choices),
                "complete_solutions": sum(
                    r["verified"] for r in games if r["strategy"] == "complete"
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
