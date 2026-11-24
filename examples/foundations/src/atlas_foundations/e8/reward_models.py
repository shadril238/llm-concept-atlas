from math import exp, log1p

chosen, rejected = 1.2, 0.4
margin = chosen - rejected
probability = 1 / (1 + exp(-margin))
loss = max(0, -margin) + log1p(exp(-abs(margin)))
print("preference probability:", round(probability, 4))
print("reward ranking loss:", round(loss, 4))
assert abs(((chosen + 100) - (rejected + 100)) - margin) < 1e-12
print("Common score offsets do not change a pairwise preference.")
