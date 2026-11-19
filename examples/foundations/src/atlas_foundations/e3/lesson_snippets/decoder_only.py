ids = ["A", "B", "C"]
for i, token in enumerate(ids):
    print(token, "can use", ids[: i + 1], "and predicts the next token")
