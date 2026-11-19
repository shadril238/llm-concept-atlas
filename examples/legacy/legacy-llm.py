from math import exp
logits = {'blue': 2.0, 'clear': 1.0, 'other': 0.0}
normalizer = sum(exp(x) for x in logits.values())
probs = {word: round(exp(x)/normalizer, 3) for word,x in logits.items()}
print(probs)
