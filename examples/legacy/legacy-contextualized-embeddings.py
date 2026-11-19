from math import sqrt
a, b = (0.8, 0.6), (0.7, 0.7)
dot = sum(x*y for x, y in zip(a, b))
cosine = dot / (sqrt(sum(x*x for x in a)) * sqrt(sum(y*y for y in b)))
print(round(cosine, 3))
