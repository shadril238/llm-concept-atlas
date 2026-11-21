steps = [(7, 8, 14), (7, 8, 15)]
for left, right, claimed in steps:
    print(f"{left} + {right} = {claimed}", "verified:", left + right == claimed)
assert 7 + 8 == 15
