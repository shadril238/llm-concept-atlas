from math import exp, log

positive, negative, temperature = 2.0, 0.0, 1.0
p = exp(positive / temperature) / (
    exp(positive / temperature) + exp(negative / temperature)
)
print("positive probability:", round(p, 3), "loss:", round(-log(p), 3))
