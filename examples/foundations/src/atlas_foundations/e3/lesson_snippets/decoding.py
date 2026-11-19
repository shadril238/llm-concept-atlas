from math import exp

logits = [2.0, 1.0, 0.0]
p = [exp(x - max(logits)) for x in logits]
p = [x / sum(p) for x in p]
kept = [0, 1]
print(
    "top-p .8:",
    [round(p[i] / sum(p[j] for j in kept), 3) if i in kept else 0 for i in range(3)],
)
