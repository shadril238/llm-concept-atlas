from math import sqrt
vectors = {'query': (0.8, 0.5), 'electricity tip': (0.7, 0.6), 'energy drink': (-0.2, 0.9)}
def cosine(a,b):
    return sum(x*y for x,y in zip(a,b))/(sqrt(sum(x*x for x in a))*sqrt(sum(y*y for y in b)))
for name, vector in vectors.items():
    if name != 'query': print(name, round(cosine(vectors['query'],vector),2))
