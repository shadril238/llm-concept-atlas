from math import log, exp
chosen_logprob, rejected_logprob = -0.4, -1.1
reference_gap = 0.2
beta = 0.5
gap = beta * ((chosen_logprob - rejected_logprob) - reference_gap)
loss = log(1 + exp(-gap))
print(round(loss, 3))
