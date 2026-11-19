"""An exact rank-one update with the lesson's column-vector convention."""

# region demonstration
W = [[1.0, 0.0], [0.0, 1.0]]  # [d_out=2, d_in=2]
A = [[0.3, -0.1]]  # [r=1, d_in=2]
B = [[0.1], [0.2]]  # [d_out=2, r=1]
x = [2.0, 1.0]
alpha, rank = 1.0, 1
compressed = sum(a * value for a, value in zip(A[0], x))
update = [(alpha / rank) * row[0] * compressed for row in B]
output = [
    sum(w * value for w, value in zip(row, x)) + delta for row, delta in zip(W, update)
]
assert all(abs(a - b) < 1e-12 for a, b in zip(output, [2.05, 1.10]))
print("A @ x:", round(compressed, 2))
print("update:", [round(value, 2) for value in update])
print("output:", [round(value, 2) for value in output])
# endregion demonstration
