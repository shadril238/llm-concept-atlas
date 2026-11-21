"""Small, explicit E5 output and tool contracts; no model or network dependency."""

from collections import Counter
import json
import math
import re

ORDER_SCHEMA = {
    "type": "object",
    "properties": {
        "item": {"type": "string", "enum": ["book", "pen"]},
        "quantity": {"type": "integer", "enum": list(range(1, 10))},
        "priority": {"type": "string", "enum": ["normal", "urgent"]},
    },
    "required": ["item", "quantity", "priority"],
    "additionalProperties": False,
}


def validate_order(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=unique)
    except (ValueError, TypeError):
        return {"valid": False, "value": None, "error": "not one complete JSON value"}
    if not isinstance(value, dict) or set(value) != {"item", "quantity", "priority"}:
        return {
            "valid": False,
            "value": value,
            "error": "exactly item, quantity, priority are required",
        }
    if (
        value["item"] not in ["book", "pen"]
        or value["priority"] not in ["normal", "urgent"]
        or type(value["quantity"]) is not int
        or not 1 <= value["quantity"] <= 9
    ):
        return {
            "valid": False,
            "value": value,
            "error": "wrong enum or quantity type/range",
        }
    return {"valid": True, "value": value, "error": None}


def extract_answer(raw, mode="integer"):
    pattern = (
        r"\s*(-?\d+)\s*"
        if mode == "integer"
        else r"(?:[\s\S]*\n)?[ \t]*Answer:[ \t]*(-?\d+)\s*"
    )
    match = re.fullmatch(pattern, raw)
    return int(match[1]) if match else None


def vote(values):
    if not values:
        raise ValueError("At least one sample is required")
    counts = Counter(value for value in values if type(value) is int)
    ranked = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    tied = len(ranked) > 1 and ranked[0][1] == ranked[1][1]
    return {
        "winner": ranked[0][0] if ranked and not tied else None,
        "counts": [{"answer": answer, "votes": count} for answer, count in ranked],
        "invalid": sum(type(v) is not int for v in values),
        "samples": len(values),
        "tied": tied,
        "strict_majority": bool(ranked and not tied and ranked[0][1] > len(values) / 2),
    }


def calculator(operation, left, right):
    if operation not in ["add", "subtract", "multiply", "divide"]:
        raise ValueError("Unknown operation")
    if any(
        type(x) not in [int, float] or not math.isfinite(x) or abs(x) > 1_000_000
        for x in [left, right]
    ):
        raise ValueError(
            "Finite bounded numeric operands are required; booleans are invalid"
        )
    if operation == "divide" and right == 0:
        raise ValueError("Division by zero")
    return {
        "add": lambda: left + right,
        "subtract": lambda: left - right,
        "multiply": lambda: left * right,
        "divide": lambda: left / right,
    }[operation]()


def dispatch(action, fixtures):
    if (
        not isinstance(action, dict)
        or set(action) != {"tool", "arguments"}
        or not isinstance(action["arguments"], dict)
    ):
        raise ValueError("Tool call requires exactly tool and arguments")
    arguments = action["arguments"]
    if action["tool"] == "calculator" and set(arguments) == {
        "operation",
        "left",
        "right",
    }:
        return calculator(**arguments)
    if action["tool"] == "lookup" and set(arguments) == {"key"}:
        if type(arguments["key"]) is not str or arguments["key"] not in fixtures:
            raise ValueError("Unknown fixture key")
        return fixtures[arguments["key"]]
    raise ValueError("Unknown tool or invalid argument fields")


def bounded_loop(choose, execute, question, max_steps=3):
    if type(max_steps) is not int or not 1 <= max_steps <= 10:
        raise ValueError("max_steps must be an integer from 1 to 10")
    trace = []
    for step in range(max_steps):
        raw = choose(question, trace)
        try:
            action = json.loads(raw)
            if (
                isinstance(action, dict)
                and set(action) == {"final"}
                and type(action["final"]) is str
            ):
                trace.append(
                    {
                        "step": step,
                        "raw": raw,
                        "kind": "final",
                        "answer": action["final"],
                    }
                )
                return {
                    "answer": action["final"],
                    "stop_reason": "final",
                    "trace": trace,
                }
            observation = execute(action)
            trace.append(
                {
                    "step": step,
                    "raw": raw,
                    "kind": "tool",
                    "action": action,
                    "observation": str(observation)[:160],
                }
            )
        except (ValueError, TypeError, KeyError) as error:
            trace.append(
                {
                    "step": step,
                    "raw": raw,
                    "kind": "invalid",
                    "observation": str(error)[:160],
                }
            )
    return {"answer": None, "stop_reason": "max_steps", "trace": trace}
