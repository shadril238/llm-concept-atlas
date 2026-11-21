"""Exact rational Game-24 search with bounded work and explicit branch traces."""

import ast
from collections import Counter, deque
from fractions import Fraction
import operator


def verify(expression, numbers, target=24):
    try:
        tree = ast.parse(expression, mode="eval")
        if len(list(ast.walk(tree))) > 80:
            return False
        leaves = []

        def visit(node):
            if (
                isinstance(node, ast.Constant)
                and type(node.value) is int
                and 1 <= node.value <= 9
            ):
                leaves.append(node.value)
                return Fraction(node.value)
            if isinstance(node, ast.BinOp) and type(node.op) in [
                ast.Add,
                ast.Sub,
                ast.Mult,
                ast.Div,
            ]:
                return {
                    ast.Add: operator.add,
                    ast.Sub: operator.sub,
                    ast.Mult: operator.mul,
                    ast.Div: operator.truediv,
                }[type(node.op)](visit(node.left), visit(node.right))
            raise ValueError(
                "Only declared binary arithmetic and integer leaves are allowed"
            )

        result = visit(tree.body)
        return result == target and Counter(leaves) == Counter(numbers)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return False


def moves(state):
    result = []
    for i, a in enumerate(state):
        for j, b in enumerate(state):
            if i >= j:
                continue
            for symbol, left, right in [
                ("+", a, b),
                ("*", a, b),
                ("-", a, b),
                ("-", b, a),
                ("/", a, b),
                ("/", b, a),
            ]:
                if symbol == "/" and right[0] == 0:
                    continue
                value = {
                    "+": operator.add,
                    "-": operator.sub,
                    "*": operator.mul,
                    "/": operator.truediv,
                }[symbol](left[0], right[0])
                rest = [item for k, item in enumerate(state) if k not in [i, j]]
                result.append(
                    (
                        rest + [(value, f"({left[1]}{symbol}{right[1]})")],
                        f"{left[1]} {symbol} {right[1]} = {value}",
                    )
                )
    return result


def key(state):
    return tuple(sorted(value for value, _ in state))


def search(numbers, beam=None, max_nodes=4000, ranker=None):
    if (
        len(numbers) != 4
        or any(type(n) is not int or not 1 <= n <= 9 for n in numbers)
        or type(max_nodes) is not int
        or max_nodes < 1
        or (beam is not None and (type(beam) is not int or beam < 1))
    ):
        raise ValueError(
            "Four integers 1–9, positive node budget, and optional positive beam required"
        )
    states = [[(Fraction(n), str(n)) for n in numbers]]
    nodes = [
        {
            "id": 0,
            "parent": None,
            "depth": 0,
            "values": [str(n) for n in numbers],
            "action": "start",
            "status": "frontier",
        }
    ]
    frontier = [0]
    seen = {key(states[0])}
    expanded = duplicate_pruned = beam_pruned = 0
    solution = None
    stop = "frontier_exhausted"
    for depth in range(4):
        following = []
        for index in frontier:
            state = states[index]
            if len(state) == 1 and state[0][0] == 24:
                nodes[index]["status"] = "solution"
                solution = state[0][1]
                stop = "solution"
                break
            if len(state) == 1:
                nodes[index]["status"] = "dead_end"
                continue
            nodes[index]["status"] = "expanded"
            expanded += 1
            candidates = moves(state)
            candidates.sort(
                key=lambda row: (
                    min(abs(v - 24) for v, _ in row[0]),
                    sum(v.denominator != 1 for v, _ in row[0]),
                    row[1],
                )
            )
            if ranker and candidates:
                chosen = ranker(candidates[:8], nodes[index])
                if type(chosen) is not int or not 0 <= chosen < min(8, len(candidates)):
                    raise ValueError("Ranker returned an unavailable candidate")
                candidates.insert(0, candidates.pop(chosen))
            for position, (next_state, action) in enumerate(candidates):
                signature = key(next_state)
                if signature in seen:
                    duplicate_pruned += 1
                    continue
                if len(nodes) >= max_nodes:
                    stop = "node_budget"
                    break
                seen.add(signature)
                states.append(next_state)
                child = len(nodes)
                nodes.append(
                    {
                        "id": child,
                        "parent": index,
                        "depth": depth + 1,
                        "values": [str(v) for v, _ in next_state],
                        "action": action,
                        "status": "frontier",
                        "model_preferred": bool(ranker and position == 0),
                    }
                )
                following.append(child)
            if stop == "node_budget":
                break
        if solution or stop == "node_budget":
            break
        if beam is not None and len(following) > beam:
            following.sort(
                key=lambda index: (
                    not nodes[index].get("model_preferred", False) if ranker else False,
                    min(abs(v - 24) for v, _ in states[index]),
                    sum(v.denominator != 1 for v, _ in states[index]),
                    nodes[index]["action"],
                )
            )
            for index in following[beam:]:
                nodes[index]["status"] = "beam_pruned"
            beam_pruned += len(following) - beam
            following = following[:beam]
        frontier = following
        if not frontier:
            break
    if solution:
        assert verify(solution, numbers)
    elif beam is not None and stop == "frontier_exhausted":
        stop = "beam_exhausted"
    # The display is explicitly a bounded subset; complete traces remain local.
    solution_node = next((n["id"] for n in nodes if n["status"] == "solution"), None)
    path = []
    while solution_node is not None:
        path.append(solution_node)
        solution_node = nodes[solution_node]["parent"]
    display_ids = set(range(min(100, len(nodes)))) | set(path)
    return {
        "numbers": numbers,
        "beam": beam,
        "max_nodes": max_nodes,
        "solution": solution,
        "verified": bool(solution and verify(solution, numbers)),
        "stop_reason": stop,
        "expanded": expanded,
        "created": len(nodes),
        "duplicate_pruned": duplicate_pruned,
        "beam_pruned": beam_pruned,
        "solution_path": list(reversed(path)),
        "display_nodes": [n for n in nodes if n["id"] in display_ids],
        "display_truncated": len(display_ids) < len(nodes),
        "full_trace": nodes,
    }
