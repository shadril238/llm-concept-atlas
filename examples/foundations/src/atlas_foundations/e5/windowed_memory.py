history = [("u1", "a1"), ("u2", "a2"), ("u3", "a3"), ("u4", "a4")]
k = 2
kept = history[-k:]
omitted = history[:-k]
print("kept pairs:", kept)
print("policy omissions:", omitted)
assert len(kept) == 2
