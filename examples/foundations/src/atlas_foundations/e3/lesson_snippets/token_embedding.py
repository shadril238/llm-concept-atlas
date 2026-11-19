table = [[0.1, 0.9], [0.5, 0.2], [0.8, 0.3], [0.0, 0.0]]
ids = [2, 0, 2]
vectors = [table[index] for index in ids]
print(vectors)
assert vectors[0] == vectors[2]
