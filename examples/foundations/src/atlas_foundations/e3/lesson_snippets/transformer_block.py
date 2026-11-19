from math import sqrt

x = [1.0, 2.0, 3.0]
mean = sum(x) / len(x)
ln = [(v - mean) / sqrt(sum((z - mean) ** 2 for z in x) / len(x) + 1e-5) for v in x]
rms = [v / sqrt(sum(z * z for z in x) / len(x) + 1e-5) for v in x]
print("LayerNorm:", [round(v, 3) for v in ln])
print("RMSNorm:", [round(v, 3) for v in rms])
