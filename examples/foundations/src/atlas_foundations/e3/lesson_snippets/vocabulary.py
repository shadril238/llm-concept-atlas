table = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
ids = [0, 1, 0]
print("table shape:", (len(table), len(table[0])))
print("selected rows:", [table[index] for index in ids])
