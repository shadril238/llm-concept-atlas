from math import exp

logits = [1.0, 2.0]
weights = [exp(v - max(logits)) for v in logits]
print("two-label relative scores:", [round(v / sum(weights), 3) for v in weights])
three = [exp(v - 2) for v in [1.0, 2.0, 2.0]]
print("with another candidate:", [round(v / sum(three), 3) for v in three])
