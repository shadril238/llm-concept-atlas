length = 3
mask = [[int(j <= i) for j in range(length)] for i in range(length)]
for row in mask:
    print(row)
assert mask[0][2] == 0
