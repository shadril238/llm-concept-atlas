prompt_ids = [10, 20, 30]
generated_ids = [10, 20, 30, 40, 50]
assert generated_ids[: len(prompt_ids)] == prompt_ids
print("new IDs:", generated_ids[len(prompt_ids) :])
