base = [[1.0, 0.0], [0.0, 1.0]]
a = [[0.1], [0.2]]
b = [[0.3, -0.1]]
adapted = [[round(base[i][j] + a[i][0]*b[0][j], 3) for j in range(2)] for i in range(2)]
print(adapted)
