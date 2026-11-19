from math import exp
scores = [1.2, 0.4, -0.3]
weights = [exp(x) for x in scores]
weights = [x / sum(weights) for x in weights]
values = [2.0, 0.0, 1.0]
print(round(sum(w*v for w, v in zip(weights, values)), 3))
