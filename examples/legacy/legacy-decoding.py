from random import Random
options = ['cat','dog','bird']
weights = [0.6,0.3,0.1]
print('greedy:', options[weights.index(max(weights))])
print('sampled:', Random(7).choices(options, weights=weights)[0])
