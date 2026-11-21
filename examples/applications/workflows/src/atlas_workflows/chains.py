"""L11a: plain Python first, then the same local model through current Runnables."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas_workflows.runtime import LocalModel, fixture, messages, render, save
from atlas_foundations.e5.contracts import ORDER_SCHEMA, validate_order
from jinja2.sandbox import ImmutableSandboxedEnvironment
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def model_adapter(model, schema=None, max_tokens=64):
    def invoke(value):
        roles = {"human": "user", "ai": "assistant", "system": "system"}
        converted = [
            {"role": roles[m.type], "content": m.content} for m in value.to_messages()
        ]
        return model.generate(converted, schema=schema, max_tokens=max_tokens)["raw"]

    return RunnableLambda(invoke)


def checked_order(raw):
    checked = validate_order(raw)
    if not checked["valid"]:
        raise ValueError(checked["error"])
    return checked["value"]


def main():
    model = LocalModel()
    data = fixture()
    system = "Return only one integer."
    question = data["arithmetic"][4]["question"]
    plain = model.generate(messages(question, system), max_tokens=32)
    prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", "{question}")]
    )
    chain = prompt | model_adapter(model, max_tokens=32) | StrOutputParser()
    framework_raw = chain.invoke({"question": question})
    framework_call = model.calls[-1]
    assert plain["rendered_prompt"] == framework_call["rendered_prompt"]
    assert plain["raw"] == framework_raw
    template = ImmutableSandboxedEnvironment().from_string(
        model.metadata["chat_template"]
    )
    rendered_reference = template.render(
        messages=plain["messages"], add_generation_prompt=True
    )
    assert rendered_reference == render(plain["messages"])
    assert (
        int(model.metadata["general_file_type"]) == 7
    )  # GGUF mostly-Q8_0, not an FP16 filename guess.

    # A real first model output becomes input to a real second prompt.
    row = next(r for r in data["extraction"] if r["split"] == "evaluation")
    extract_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Extract the current order into item, quantity, priority. item is book or pen; quantity is integer 1..9; priority is normal or urgent. Return only JSON.",
            ),
            ("human", "{request}"),
        ]
    )
    extract_chain = (
        extract_prompt
        | model_adapter(model, schema=ORDER_SCHEMA, max_tokens=96)
        | RunnableLambda(checked_order)
    )
    prices = {"book": 8, "pen": 2}

    def quote(payload):
        extracted = payload["extracted"]
        return dict(
            payload, calculated_total=extracted["quantity"] * prices[extracted["item"]]
        )

    quote_schema = {
        "type": "object",
        "properties": {"total": {"type": "integer", "enum": list(range(1, 73))}},
        "required": ["total"],
        "additionalProperties": False,
    }
    quote_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Copy the supplied calculated total into one JSON object with key total. Do not recalculate or change the number.",
            ),
            (
                "human",
                "Order extraction: {extracted}. Calculated total: {calculated_total}.",
            ),
        ]
    )
    pipeline = RunnablePassthrough.assign(extracted=extract_chain) | RunnableLambda(
        quote
    )
    state = pipeline.invoke({"request": row["text"]})
    first_call = model.calls[-1]
    final_raw = (
        quote_prompt
        | model_adapter(model, schema=quote_schema, max_tokens=24)
        | StrOutputParser()
    ).invoke(state)
    final = json.loads(final_raw)
    expected_total = row["expected"]["quantity"] * prices[row["expected"]["item"]]
    result = {
        "model": model.metadata,
        "plain": plain,
        "framework": framework_call,
        "equivalent_prompt_checked": True,
        "equivalent_output_checked": True,
        "gguf_template_checked": True,
        "quantization_type_checked": True,
        "multi_prompt": {
            "source_id": row["id"],
            "request": row["text"],
            "expected_extraction": row["expected"],
            "state": state,
            "first_call": first_call,
            "second_call": model.calls[-1],
            "final": final,
            "expected_total": expected_total,
            "extraction_correct": state["extracted"] == row["expected"],
            "copy_consistent": final.get("total") == state["calculated_total"],
            "end_to_end_correct": final.get("total") == expected_total,
        },
        "modernization": [
            "langchain_core ChatPromptTemplate, RunnableLambda, RunnablePassthrough.assign, and StrOutputParser replace book-era LLMChain imports.",
            "An explicitly implemented local GGUF adapter converts typed roles; no provider credentials or cloud tracing.",
            "GGUF is a container; this actual artifact reports file type 7/Q8_0. Book text discussing eight-bit loading links an FP16 filename, so format and quantization are separated here.",
        ],
        "limitations": [
            "The 135M model can fail simple arithmetic despite a valid prompt and grammar.",
            "A correct deterministic total for a wrong extraction is still wrong for the original order.",
            "This is a small declared local Runnable workflow, not an implementation of every current LangChain agent harness.",
        ],
    }
    save("e5-chains", result)
    print(
        json.dumps(
            {
                "equivalent_prompt_checked": True,
                "equivalent_output_checked": True,
                "gguf_template_checked": True,
                "quantization_type_checked": True,
                "multi_prompt_correct": result["multi_prompt"]["end_to_end_correct"],
            }
        )
    )


if __name__ == "__main__":
    main()
