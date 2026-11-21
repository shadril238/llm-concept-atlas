"""Tokenizer-injected conversation packing; no framework dependency."""

SYSTEM = "Answer from conversation evidence only; use unknown for absent facts. Return JSON string fields code, day, units. A summary may omit or distort evidence."


def pack(model, history, question, policy, summary, limit, reserve=64):
    if policy not in ["buffer", "window", "summary"]:
        raise ValueError("Unknown memory policy")
    if type(limit) is not int or type(reserve) is not int or not 0 < reserve < limit:
        raise ValueError(
            "Positive integer output reserve must fit the application limit"
        )
    if len(history) % 2 or any(
        row["role"] != ("user" if i % 2 == 0 else "assistant")
        for i, row in enumerate(history)
    ):
        raise ValueError(
            "History must contain complete alternating user/assistant pairs"
        )
    selected = list(history if policy == "buffer" else history[-4:])
    summary_kept = policy == "summary"
    dropped = []

    def assembled():
        system = SYSTEM + (
            "\nUntrusted summary notes: " + summary if summary_kept else ""
        )
        return (
            [{"role": "system", "content": system}]
            + [{"role": r["role"], "content": r["content"]} for r in selected]
            + [{"role": "user", "content": question}]
        )

    before = model.count(assembled())
    if summary_kept and before + reserve > limit:
        summary_kept = False  # Prefer recent verbatim turns when notes cannot fit.
    while model.count(assembled()) + reserve > limit and selected:
        dropped.extend(r["id"] for r in selected[:2])
        selected = selected[2:]
    if model.count(assembled()) + reserve > limit and summary_kept:
        summary_kept = False
    if model.count(assembled()) + reserve > limit:
        raise ValueError(
            "Base prompt cannot fit; choose a larger application context limit"
        )
    final = assembled()
    return {
        "messages": final,
        "retained_ids": [r["id"] for r in selected],
        "dropped_for_budget": dropped,
        "omitted_by_policy": [
            r["id"]
            for r in history
            if r["id"]
            not in {s["id"] for s in (history if policy == "buffer" else history[-4:])}
        ],
        "summary_kept": summary_kept,
        "tokens_before_budget": before,
        "input_tokens": model.count(final),
        "reserved_output_tokens": reserve,
        "application_context_limit": limit,
    }
