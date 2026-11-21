"""L11c: real model choices inside a bounded controller, plus explicit fake tests."""

import json
from pathlib import Path
import sys
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas_workflows.runtime import LocalModel, fixture, messages, save
from atlas_foundations.e5.contracts import bounded_loop, calculator, dispatch
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field, StrictInt


class CalculatorArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    operation: Literal["add", "subtract", "multiply", "divide"]
    left: StrictInt = Field(ge=-1_000_000, le=1_000_000)
    right: StrictInt = Field(ge=-1_000_000, le=1_000_000)


class LookupArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    key: Literal["library_closes", "workshop_day", "servings"]


def framework_dispatcher(fixtures):
    def lookup(key):
        return fixtures[key]

    tools = {
        "calculator": StructuredTool.from_function(
            calculator,
            name="calculator",
            description="Four explicit arithmetic operations on bounded integer operands.",
            args_schema=CalculatorArguments,
        ),
        "lookup": StructuredTool.from_function(
            lookup,
            name="lookup",
            description="Read a named value from the original local fixture only.",
            args_schema=LookupArguments,
        ),
    }

    def execute(action):
        if (
            not isinstance(action, dict)
            or set(action) != {"tool", "arguments"}
            or action["tool"] not in tools
            or not isinstance(action["arguments"], dict)
        ):
            raise ValueError("Unsupported tool call")
        return tools[action["tool"]].invoke(action["arguments"])

    return execute


def main():
    data = fixture()
    fixtures = data["tool_fixtures"]
    execute_plain = lambda action: dispatch(action, fixtures)
    execute_framework = framework_dispatcher(fixtures)
    scripts = {
        "success": [
            json.dumps({"tool": "lookup", "arguments": {"key": "library_closes"}}),
            json.dumps({"final": "17:30"}),
        ],
        "repair": [
            json.dumps(
                {
                    "tool": "calculator",
                    "arguments": {"operation": "add", "left": True, "right": 8},
                }
            ),
            json.dumps({"tool": "lookup", "arguments": {"key": "library_closes"}}),
            json.dumps({"final": "17:30"}),
        ],
        "loop": [json.dumps({"tool": "lookup", "arguments": {"key": "servings"}})] * 3,
    }
    fake_cases = []
    for name, script in scripts.items():
        stream = iter(script)
        plain = bounded_loop(
            lambda q, trace: next(stream), execute_plain, "Find library closing time", 3
        )
        fake_model = FakeListLLM(responses=script)
        choose = lambda q, trace: fake_model.invoke(q + "\n" + json.dumps(trace))
        framework = RunnableLambda(
            lambda payload: bounded_loop(
                choose, execute_framework, payload["question"], 3
            )
        ).invoke({"question": "Find library closing time"})
        # Error wording can differ between explicit and Pydantic validation.
        assert [r["kind"] for r in plain["trace"]] == [
            r["kind"] for r in framework["trace"]
        ]
        assert (
            plain["answer"] == framework["answer"]
            and plain["stop_reason"] == framework["stop_reason"]
        )
        fake_cases.append(
            {
                "id": name,
                "path": "scripted fake-model harness test; no model competence claim",
                "plain": plain,
                "framework": framework,
            }
        )
    assert fake_cases[-1]["framework"]["stop_reason"] == "max_steps"

    model = LocalModel()
    real_cases = []
    for task in data["tool_tasks"]:
        for engine, execute in [
            ("plain", execute_plain),
            ("langchain", execute_framework),
        ]:
            decisions = []

            def choose(question, trace):
                if not trace:
                    candidates = {
                        "lookup-" + key: {"tool": "lookup", "arguments": {"key": key}}
                        for key in fixtures
                    }
                    if "left" in task:
                        candidates.update(
                            {
                                operation: {
                                    "tool": "calculator",
                                    "arguments": {
                                        "operation": operation,
                                        "left": task["left"],
                                        "right": task["right"],
                                    },
                                }
                                for operation in [
                                    "add",
                                    "subtract",
                                    "multiply",
                                    "divide",
                                ]
                            }
                        )
                    prompt = f"Choose the best available tool action for the question. Return only its JSON string ID.\nQuestion: {question}\nActions: {json.dumps(candidates)}"
                else:
                    # This domain is predeclared, never the case's expected answer alone.
                    answers = list(
                        dict.fromkeys(
                            [str(i) for i in range(101)]
                            + list(fixtures.values())
                            + ["unknown"]
                        )
                    )
                    candidates = {answer: {"final": answer} for answer in answers}
                    prompt = f"Answer the question using the actual tool observation. Return only the answer as a JSON string.\nQuestion: {question}\nObservation: {json.dumps(trace[-1])}"
                call = model.generate(
                    messages(prompt),
                    max_tokens=20,
                    schema={"type": "string", "enum": list(candidates)},
                )
                selected = json.loads(call["raw"])
                assert selected in candidates
                decisions.append(
                    {
                        "step": len(trace),
                        "choice_ids": list(candidates),
                        "selected_id": selected,
                        "resolved_action": candidates[selected],
                        "call": call,
                    }
                )
                return json.dumps(candidates[selected])

            if engine == "plain":
                outcome = bounded_loop(choose, execute, task["question"], 3)
            else:
                outcome = RunnableLambda(
                    lambda payload: bounded_loop(
                        choose, execute, payload["question"], 3
                    )
                ).invoke({"question": task["question"]})
            assert len(outcome["trace"]) <= 3
            real_cases.append(
                {
                    "id": task["id"],
                    "engine": engine,
                    "question": task["question"],
                    "expected": task["expected"],
                    "outcome": outcome,
                    "correct": outcome["answer"] == task["expected"],
                    "decisions": decisions,
                }
            )
    result = {
        "model": model.metadata,
        "fixtures": fixtures,
        "fake_cases": fake_cases,
        "real_cases": real_cases,
        "termination_checked": True,
        "invalid_arguments_checked": True,
        "framework_tools_executed": True,
        "policy": {
            "max_steps": 3,
            "observation_character_cap": 160,
            "required_tool_before_final": True,
            "operand_source": "Explicit versioned task fields; model selects operation/tool, not arbitrary operand extraction",
            "final_domain": "Integers 0..100, all fixture values, unknown; same predeclared domain for every case",
            "tools": ["bounded integer calculator", "exact local fixture lookup"],
        },
        "limitations": [
            "A small model controls a restricted choice inside an explicit state machine, not an unrestricted autonomous agent.",
            "The controller resolves a generated choice ID to a validated call; actual model IDs and resolved arguments are shown separately.",
            "Native grammar enforces choice format, not correct tool selection or answer.",
            "The fake-model tests verify harness behavior and are labeled separately from real model execution.",
            "No arbitrary eval, shell, browser, private file, or write tool is available. Tool observations are data, not authority to expand permissions.",
        ],
    }
    save("e5-agents", result)
    print(
        json.dumps(
            {
                "termination_checked": True,
                "invalid_arguments_checked": True,
                "framework_tools_executed": True,
                "real_cases": len(real_cases),
                "correct": sum(r["correct"] for r in real_cases),
            }
        )
    )


if __name__ == "__main__":
    main()
