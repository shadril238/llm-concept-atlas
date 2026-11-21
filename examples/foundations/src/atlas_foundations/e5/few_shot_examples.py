examples = [{"id": "design-add", "answer": 5}, {"id": "design-sub", "answer": 5}]
print("example IDs:", [row["id"] for row in examples])
print("distinct demonstrated answers:", len({row["answer"] for row in examples}))
print("reversed IDs:", [row["id"] for row in reversed(examples)])
