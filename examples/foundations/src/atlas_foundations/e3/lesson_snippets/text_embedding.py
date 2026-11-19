states = [[1.0, 0.0], [0.0, 1.0], [9.0, 9.0]]
mask = [1, 1, 0]
pooled = [
    sum(row[d] * keep for row, keep in zip(states, mask)) / sum(mask) for d in range(2)
]
print("masked mean:", pooled)
