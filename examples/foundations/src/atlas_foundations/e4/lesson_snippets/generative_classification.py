outputs = [" negative ", "POSITIVE", "not negative", "positive because it is funny"]
label_ids = {"negative": 0, "positive": 1}
parsed = [label_ids.get(raw.strip().casefold(), -1) for raw in outputs]
print(parsed)
assert parsed == [0, 1, -1, -1]
