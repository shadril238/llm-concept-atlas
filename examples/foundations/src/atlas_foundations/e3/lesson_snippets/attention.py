from math import exp

scores, values = [2.0, 1.0], [10.0, 0.0]
p = [exp(s - max(scores)) for s in scores]
p = [x / sum(p) for x in p]
print("weights:", [round(x, 3) for x in p])
print("output:", round(sum(w * v for w, v in zip(p, values)), 3))
