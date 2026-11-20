text = "not good"
positive_words = {"good", "excellent", "wonderful"}
baseline = int(bool(set(text.split()) & positive_words))
gold = 0
print("keyword prediction:", baseline, "policy label:", gold)
assert baseline != gold  # a deliberately exposed baseline failure
