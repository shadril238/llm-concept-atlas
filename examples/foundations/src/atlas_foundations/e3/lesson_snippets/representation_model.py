from math import sqrt

a, b = [1.0, 2.0], [2.0, 4.0]
cosine = sum(x * y for x, y in zip(a, b)) / sqrt(
    sum(x * x for x in a) * sum(x * x for x in b)
)
print("same direction:", round(cosine, 3))
