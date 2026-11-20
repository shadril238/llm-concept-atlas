from math import exp

query = [0.6, 0.8]
prototypes = [[1.0, 0.0], [0.0, 1.0]]
scores = [sum(a * b for a, b in zip(query, p)) for p in prototypes]
z = sum(a * b for a, b in zip(query, [2.0, -1.0]))
print("prototype scores:", scores)
print("linear-head positive probability:", round(1 / (1 + exp(-z)), 3))
