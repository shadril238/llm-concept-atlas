from math import sqrt
query = (0.9,0.3)
docs = {'power-saving tips':(0.8,0.4), 'billing policy':(0.4,0.8)}
def cos(a,b):
    return sum(x*y for x,y in zip(a,b))/(sqrt(sum(x*x for x in a))*sqrt(sum(y*y for y in b)))
print(sorted(docs, key=lambda name: cos(query,docs[name]), reverse=True))
