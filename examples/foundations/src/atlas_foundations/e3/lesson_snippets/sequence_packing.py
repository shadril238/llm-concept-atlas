segments = [0, 0, 1, 1]
mask = [
    [int(j <= i and segments[i] == segments[j]) for j in range(4)] for i in range(4)
]
for row in mask:
    print(row)
assert mask[2][0] == 0 and mask[3][2] == 1
