"""A stable softmax and weighted sum, using invented scalar values."""
# region demonstration
from math import exp, isclose, isfinite


def attend(scores: list[float], values: list[float]) -> tuple[list[float], float]:
    if not scores or len(scores) != len(values):
        raise ValueError("Provide equal nonempty score and value lists")
    if not all(isfinite(x) for x in scores + values):
        raise ValueError("Scores and values must be finite")
    # Subtracting the maximum preserves softmax and prevents overflow.
    shifted = [exp(score - max(scores)) for score in scores]
    weights = [value / sum(shifted) for value in shifted]
    return weights, sum(weight * value for weight, value in zip(weights, values))


def main() -> None:
    weights, output = attend([2.0, 1.0], [10.0, 0.0])
    assert isclose(sum(weights), 1.0)
    assert isclose(output, 7.3105857863, abs_tol=1e-9)
    shifted_weights, _ = attend([1002.0, 1001.0], [10.0, 0.0])
    assert all(isclose(a, b) for a, b in zip(weights, shifted_weights))
    print("weights:", [round(weight, 4) for weight in weights])
    print("weighted value:", round(output, 4))


if __name__ == "__main__":
    main()
# endregion demonstration
