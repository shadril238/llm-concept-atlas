positions = range(3)
print("encoder:", [[1 for j in positions] for i in positions])
print("causal:", [[int(j <= i) for j in positions] for i in positions])
