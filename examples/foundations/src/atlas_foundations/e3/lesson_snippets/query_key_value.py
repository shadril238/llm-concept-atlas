from math import exp, sqrt

query, keys, values = [1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], [[10.0, 0.0], [0.0, 4.0]]
scores = [sum(a * b for a, b in zip(query, key)) / sqrt(2) for key in keys]
p = [exp(s - max(scores)) for s in scores]
p = [x / sum(p) for x in p]
print([round(sum(p[j] * values[j][d] for j in range(2)), 3) for d in range(2)])
