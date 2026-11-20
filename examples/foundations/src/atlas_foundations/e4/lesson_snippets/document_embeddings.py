from math import sqrt

states = [[2.0, 0.0], [0.0, 2.0], [100.0, 100.0]]
mask = [1, 1, 0]
mean = [sum(row[j] * m for row, m in zip(states, mask)) / sum(mask) for j in range(2)]
norm = sqrt(sum(v * v for v in mean))
print("masked mean:", mean, "unit vector:", [round(v / norm, 3) for v in mean])
