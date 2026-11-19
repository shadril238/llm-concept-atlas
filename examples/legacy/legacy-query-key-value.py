from math import exp
query, keys, values = 2, [1,3], [10,0]
scores = [query*key for key in keys]
weights = [exp(s)/sum(exp(x) for x in scores) for s in scores]
print(round(sum(w*v for w,v in zip(weights,values)),2))
