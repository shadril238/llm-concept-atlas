"""L11b: actual summary updates, token budgets, and three memory policies."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas_workflows.runtime import LocalModel, fixture, messages, save
from atlas_workflows.chains import model_adapter
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {key: {"type": "string"} for key in ["code", "day", "units"]},
    "required": ["code", "day", "units"],
    "additionalProperties": False,
}
from atlas_foundations.e5.memory import SYSTEM, pack


def main():
    model = LocalModel()
    data = fixture()
    history = data["conversation"]
    summary = ""
    updates = []
    for offset in [0, 4]:
        chunk = history[offset : offset + 4]
        prompt = (
            f"Update concise conversation notes from the existing notes and new messages. Preserve exact codes, unit preferences, and the latest correction to the day.\nExisting notes: {summary or 'none'}\nNew messages:\n"
            + "\n".join(f"{r['role']}: {r['content']}" for r in chunk)
        )
        call = model.generate(messages(prompt), max_tokens=80)
        updates.append(
            {
                "prior_summary": summary,
                "new_message_ids": [r["id"] for r in chunk],
                "call": call,
                "summary": call["raw"],
            }
        )
        summary = call["raw"]
    cases = []
    for limit in [192, 384]:
        for policy in ["buffer", "window", "summary"]:
            packed = pack(
                model, history, data["memory_question"], policy, summary, limit
            )
            call = model.generate(
                packed["messages"], max_tokens=64, schema=ANSWER_SCHEMA
            )
            assert call["prompt_tokens"] + 64 <= limit
            try:
                parsed = json.loads(call["raw"])
            except json.JSONDecodeError:
                parsed = {}
            scores = {
                key: isinstance(parsed, dict)
                and type(parsed.get(key)) is str
                and parsed[key].strip().casefold() == expected.casefold()
                for key, expected in data["memory_facts"].items()
            }
            evidence_text = "\n".join(
                r["content"] for r in packed["messages"][:-1]
            ).casefold()
            presence = {
                key: expected.casefold() in evidence_text
                for key, expected in data["memory_facts"].items()
            }
            cases.append(
                {
                    "policy": policy,
                    "budget": limit,
                    "packed": packed,
                    "call": call,
                    "parsed": parsed,
                    "field_correct": scores,
                    "literal_evidence_presence": presence,
                    "stale_friday_present": "friday" in evidence_text,
                }
            )
    # Framework equivalent uses explicit application history, not a deprecated memory class.
    reference = next(r for r in cases if r["policy"] == "window" and r["budget"] == 384)
    store = [
        HumanMessage(content=r["content"])
        if r["role"] == "user"
        else AIMessage(content=r["content"])
        for r in reference["packed"]["messages"][1:-1]
    ]
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM), MessagesPlaceholder("history"), ("human", "{question}")]
    )
    framework_raw = (
        prompt | model_adapter(model, schema=ANSWER_SCHEMA, max_tokens=64)
    ).invoke({"history": store, "question": data["memory_question"]})
    assert model.calls[-1]["rendered_prompt"] == reference["call"]["rendered_prompt"]
    assert framework_raw == reference["call"]["raw"]
    result = {
        "model": model.metadata,
        "history": history,
        "question": data["memory_question"],
        "expected_facts": data["memory_facts"],
        "summary_updates": updates,
        "cases": cases,
        "answer_schema": ANSWER_SCHEMA,
        "framework_equivalent_checked": True,
        "framework_call": model.calls[-1],
        "budget_accounting_checked": True,
        "limitations": [
            "Memory is application-retained prompt data; no model weights change.",
            "A fixed two-turn window is not a token guarantee; whole turns are dropped again when required by the application budget.",
            "Summary memory here means running summary plus two recent turns; summaries are generated outputs, not an accurate database.",
            "Literal evidence presence is a text diagnostic, not proof the model used or interpreted a fact correctly.",
            "The field checker uses exact normalized strings on three declared facts; it is not general semantic evaluation.",
            "The application budgets 192/384 include reserved output; native runtime capacity is separately 768 tokens.",
        ],
    }
    save("e5-memory", result)
    print(
        json.dumps(
            {
                "framework_equivalent_checked": True,
                "budget_accounting_checked": True,
                "cases": len(cases),
                "field_scores": [
                    {
                        "policy": r["policy"],
                        "budget": r["budget"],
                        "correct": sum(r["field_correct"].values()),
                    }
                    for r in cases
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
