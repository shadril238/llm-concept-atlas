from math import exp
scores = [2.0, 1.0]
weights = [exp(x)/sum(exp(y) for y in scores) for x in scores]
values = [10.0, 0.0]
print([round(w, 2) for w in weights], round(sum(w*v for w,v in zip(weights,values)),2))
