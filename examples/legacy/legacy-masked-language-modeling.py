tokens = ["the", "model", "reads"]
target = tokens[1]
masked = tokens.copy(); masked[1] = "[MASK]"
print(masked, "target:", target)
