"""Offline hosted-grounding and optional judge import; never makes API requests."""


def import_grounded(fixture, envelope):
    required = ["provider", "model", "prompt_version", "collected_at"]
    if any(
        type(envelope.get(key)) is not str or not envelope[key].strip()
        for key in required
    ):
        raise ValueError("Provider/model/prompt/time provenance is required")
    allowed = {q["id"] for q in fixture["queries"] if q["split"] == "evaluation"}
    sources = {d["id"]: d["text"] for d in fixture["documents"]}
    outputs = {}
    for output in envelope.get("outputs", []):
        if output.get("id") not in allowed or output["id"] in outputs:
            raise ValueError("Unknown or duplicate evaluation ID")
        answer = output.get("answer")
        error = output.get("error")
        if type(answer) is not str and not error:
            raise ValueError("Output needs a string answer or explicit error")
        checked = []
        for citation in output.get("citations", []):
            id = citation.get("source_id")
            start = citation.get("answer_start")
            end = citation.get("answer_end")
            a = citation.get("source_start")
            b = citation.get("source_end")
            if id not in sources:
                raise ValueError("Citation source ID does not resolve")
            if any(type(v) is not int for v in [start, end, a, b]):
                raise ValueError(
                    "Citation offsets must be integer Unicode code-point offsets"
                )
            if (
                type(answer) is not str
                or not 0 <= start < end <= len(answer)
                or not 0 <= a < b <= len(sources[id])
            ):
                raise ValueError("Citation span is outside answer/source")
            if citation.get("quote") != sources[id][a:b]:
                raise ValueError("Quote does not match source span")
            checked.append(
                {
                    "source_id": id,
                    "claim": answer[start:end],
                    "quote": citation["quote"],
                    "span_resolved": True,
                    "semantic_support": "not judged",
                }
            )
        outputs[output["id"]] = {"answer": answer, "error": error, "citations": checked}
    return {
        "provenance": {key: envelope[key] for key in required},
        "requested": len(allowed),
        "received": len(outputs),
        "missing_ids": sorted(allowed - set(outputs)),
        "errors": sum(bool(r["error"]) for r in outputs.values()),
        "outputs": outputs,
        "execution": "External provider execution not independently verified; importer made zero API calls. Citation span resolution is not semantic support.",
    }


def import_judgments(local_rows, envelope):
    import hashlib

    if not envelope.get("judge_model") or not envelope.get("rubric_version"):
        raise ValueError("Judge provenance required")
    by_key = {(r["id"], r["variant"]): r["answer_call"]["raw"] for r in local_rows}
    seen = set()
    for judgment in envelope.get("rows", []):
        key = (judgment.get("id"), judgment.get("variant"))
        if key not in by_key or key in seen:
            raise ValueError("Unknown or duplicate judged case")
        seen.add(key)
        if hashlib.sha256(by_key[key].encode()).hexdigest() != judgment.get(
            "raw_sha256"
        ):
            raise ValueError("Judge output targets a different generated answer")
        if (
            judgment.get("support") not in ["supported", "unsupported", "uncertain"]
            or type(judgment.get("fluency")) is not int
            or not 1 <= judgment["fluency"] <= 3
        ):
            raise ValueError("Unsupported judgment labels")
    return {
        "received": len(seen),
        "unjudged": len(by_key) - len(seen),
        "status": "Externally supplied model judgments, not independent truth. Compare with source-based human review.",
        "rows": envelope.get("rows", []),
    }
