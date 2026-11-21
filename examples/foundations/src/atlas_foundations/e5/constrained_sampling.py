allowed = set(range(1, 10))
reference, generated = 3, 8
print("domain valid:", generated in allowed)
print("content correct:", generated == reference)
print("incomplete JSON:", '{"item":')
# Domain membership is an illustration; real token constraints run in L09.
