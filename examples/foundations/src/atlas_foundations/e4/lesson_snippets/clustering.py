a = [0, 0, 1, 1]
b = [7, 7, 3, 3]
pairs = [(i, j) for i in range(len(a)) for j in range(i + 1, len(a))]
agreement = sum((a[i] == a[j]) == (b[i] == b[j]) for i, j in pairs) / len(pairs)
print("raw ID equality:", sum(x == y for x, y in zip(a, b)) / len(a))
print("pair agreement:", agreement)
assert agreement == 1
