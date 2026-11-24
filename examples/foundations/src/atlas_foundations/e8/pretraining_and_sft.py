prompt_ids = [11, 12, 13]
answer_ids, eos_id, pad_count = [21, 22], 0, 2
ids = prompt_ids + answer_ids + [eos_id] + [eos_id] * pad_count
labels = [-100] * len(prompt_ids) + answer_ids + [eos_id] + [-100] * pad_count
for position, (token, label) in enumerate(zip(ids, labels)):
    print(position, token, "ignored" if label == -100 else "target")
assert labels[5] == eos_id and labels[-1] == -100
