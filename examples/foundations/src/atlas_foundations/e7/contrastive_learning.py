import sys
sys.path.insert(0, 'examples/foundations/src')
from atlas_foundations.e7.contracts import cross_entropy
scores = [0.8, 0.4]
scale = 5
logits = [scale * score for score in scores]
loss = cross_entropy(logits, 0)
from math import exp
probability = exp(-loss)
print('target probability:', round(probability, 4))
print('row loss:', round(loss, 4))
print('logit gradients:', [round(probability - 1, 4), round(1 - probability, 4)])
