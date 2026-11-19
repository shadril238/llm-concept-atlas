from math import sqrt
vectors = {'charming film': (0.9, 0.2), 'enjoyed movie': (0.8, 0.3), 'late bus': (-0.4, 0.7)}
a, b = vectors['charming film'], vectors['enjoyed movie']
cosine = sum(x*y for x,y in zip(a,b)) / (sqrt(sum(x*x for x in a))*sqrt(sum(y*y for y in b)))
print(round(cosine, 3))
