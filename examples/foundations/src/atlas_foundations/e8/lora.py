w = [[1.0, 0.0], [0.0, 1.0]]
a, b, scale, x = [0.1, 0.2], [0.3, -0.1], 2.0, [2.0, -1.0]
merged = [[w[i][j] + scale * b[i] * a[j] for j in range(2)] for i in range(2)]
separate = [
    sum(w[i][j] * x[j] for j in range(2))
    + scale * b[i] * sum(a[j] * x[j] for j in range(2))
    for i in range(2)
]
combined = [sum(merged[i][j] * x[j] for j in range(2)) for i in range(2)]
print("merged:", merged)
assert all(abs(a - b) < 1e-12 for a, b in zip(separate, combined))
print("same linear function:", combined)
