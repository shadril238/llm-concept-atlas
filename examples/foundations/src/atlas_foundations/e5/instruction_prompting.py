import re

expected = 15
for raw in ["15", "14", "Answer: 15"]:
    valid = re.fullmatch(r"-?\d+", raw) is not None
    normalized = int(raw.removeprefix("Answer: "))
    print(raw, "format:", valid, "answer:", normalized == expected)
