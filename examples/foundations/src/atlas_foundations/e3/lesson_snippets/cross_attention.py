target_queries, source_keys = 2, 3
weights = [[0.4, 0.2, 0.4], [0.2, 0.4, 0.4]]
assert len(weights) == target_queries
assert all(len(row) == source_keys and abs(sum(row) - 1) < 1e-9 for row in weights)
print("rectangular weights:", weights)
