from math import exp

logits = [2.0, 1.0, 0.0]
weights = [exp(x - max(logits)) for x in logits]
print([round(x / sum(weights), 3) for x in weights])
