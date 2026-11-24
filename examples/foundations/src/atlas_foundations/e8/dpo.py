from math import exp, log1p

pc, pr, rc, rr, beta = -3.0, -5.0, -4.0, -5.0, 0.1
margin = (pc - pr) - (rc - rr)
z = beta * margin
loss = max(0, -z) + log1p(exp(-abs(z)))
print("reference-relative margin:", margin)
print("DPO loss:", round(loss, 4))
print("No separate scalar reward model appears in this calculation.")
